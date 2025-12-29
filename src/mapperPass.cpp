/*
 * ======================================================================
 * mapperPass.cpp  (LLVM 21 ONLY)
 * ======================================================================
 * Mapper pass implementation as an LLVM New Pass Manager (NPM) plugin.
 *
 * Run with:
 *   opt-21 -load-pass-plugin ./libmapperPass.so -passes='function(mapperPass)' \
 *          -disable-output kernel.ll
 *
 * Notes:
 * - This file intentionally DROPS legacy FunctionPass/RegisterPass support.
 * - Avoids name collision with llvm::json by using alias `njson`.
 */

#include <llvm/IR/Function.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Support/raw_ostream.h>

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <set>
#include <map>
#include <list>
#include <string>
#include <cassert>
#include <chrono>

#include "json.hpp"
#include "Mapper.h"

// Used to workaround the mis-interpret of LLVM opcode in github
// testing infra: https://github.com/tancheng/CGRA-Mapper/pull/27#issuecomment-2495202802
extern int testing_opcode_offset;

using std::cout;
using std::endl;
using std::ifstream;
using std::list;
using std::map;
using std::set;
using std::string;

// IMPORTANT: avoid name collision with llvm::json namespace in LLVM 21
using njson = nlohmann::json;

void addDefaultKernels(map<string, list<int>*>*);

namespace {

static list<llvm::Loop*>* getTargetLoopsImpl(llvm::Function& t_F,
                                            map<string, list<int>*>* t_functionWithLoop,
                                            bool t_targetNested,
                                            llvm::LoopInfo &LI) {
  int targetLoopID = 0;
  auto* targetLoops = new list<llvm::Loop*>();

  // Since the ordering of the target loop id could be random, use O(n^2) to search the target loop.
  while((*t_functionWithLoop).at(t_F.getName().str())->size() > 0) {
    targetLoopID = (*t_functionWithLoop).at(t_F.getName().str())->front();
    (*t_functionWithLoop).at(t_F.getName().str())->pop_front();

    int tempLoopID = 0;
    llvm::Loop* current_loop = nullptr;

    for (auto loopItr = LI.begin(); loopItr != LI.end(); ++loopItr) {
      current_loop = *loopItr;
      if (tempLoopID == targetLoopID) {
        // Targets innermost loop if the param targetNested is not set.
        if (!t_targetNested) {
          while (!current_loop->getSubLoops().empty()) {
            llvm::errs() << "[explore] nested loop ... subloop size: "
                         << current_loop->getSubLoops().size() << "\n";
            // TODO: might change '0' to a reasonable index
            current_loop = current_loop->getSubLoops()[0];
          }
        }
        targetLoops->push_back(current_loop);
        llvm::errs() << "*** reach target loop ID: " << tempLoopID << "\n";
        break;
      }
      ++tempLoopID;
    }

    if (targetLoops->size() == 0) {
      llvm::errs() << "... no loop detected in the target kernel ...\n";
    }
  }

  llvm::errs() << "... done detected loops.size(): " << targetLoops->size() << "\n";
  return targetLoops;
}

/*
 * Early exit if mapping is not possible when no FU can support certain DFG op.
 * Lists all the missing fus.
 */
static bool canMapImpl(CGRA* t_cgra, DFG* t_dfg) {
  std::set<std::string> missing_fus;

  for (auto it = t_dfg->nodes.begin(); it != t_dfg->nodes.end(); ++it) {
    DFGNode* node = *it;
    bool nodeSupported = false;

    for (int i = 0; i < t_cgra->getRows() && !nodeSupported; ++i) {
      for (int j = 0; j < t_cgra->getColumns(); ++j) {
        CGRANode* fu = t_cgra->nodes[i][j];
        if (fu && fu->canSupport(node)) {
          nodeSupported = true;
          break;
        }
      }
    }

    if (!nodeSupported) {
      missing_fus.insert(node->getOpcodeName());
    }
  }

  if (!missing_fus.empty()) {
    std::cout << "[canMap] Missing functional units: ";
    for (const auto& op : missing_fus) {
      std::cout << op << " ";
    }
    std::cout << std::endl;
    return false;
  }

  return true;
}

/*
 * Shared implementation body for NPM.
 * Takes LoopInfo as an argument (obtained from LoopAnalysis).
 * Returns whether IR was modified (this pass does not modify IR -> false).
 */
static bool runMapperImpl(llvm::Function &t_F, llvm::LoopInfo &LI) {

  // Initializes input parameters.
  int rows                      = 4;
  int columns                   = 4;
  bool targetEntireFunction     = false;
  bool targetNested             = false;
  bool doCGRAMapping            = true;
  bool isStaticElasticCGRA      = false;
  bool isTrimmedDemo            = true;
  int ctrlMemConstraint         = 200;
  int bypassConstraint          = 4;
  int regConstraint             = 8;
  bool precisionAware           = false;
  std::string vectorizationMode = "all";
  bool heuristicMapping         = true;
  bool parameterizableCGRA      = false;

  // Incremental mapping related:
  // https://github.com/tancheng/CGRA-Mapper/pull/24
  bool incrementalMapping       = false;

  // DVFS-related options.
  bool supportDVFS              = false;
  bool DVFSAwareMapping         = false;
  int DVFSIslandDim             = 2;
  bool enablePowerGating        = false;
  bool enableExpandableMapping  = false;

  // Option used to split one integer division into 4.
  // https://github.com/tancheng/CGRA-Mapper/pull/27#issuecomment-2480362586
  int vectorFactorForIdiv       = 1;
  string multiCycleStrategy     = "exclusive";

  auto* execLatency     = new map<string, int>();
  auto* pipelinedOpt    = new list<string>();
  auto* fusionStrategy  = new list<string>();
  auto* additionalFunc  = new map<string, list<int>*>();
  auto* fusionPattern   = new map<string, list<string>*>();

  // Set the target function and loop.
  auto* functionWithLoop = new map<string, list<int>*>();
  addDefaultKernels(functionWithLoop);

  // Read the parameter JSON file.
  ifstream i("./param.json");
  if (!i.good()) {
    cout<< "=============================================================\n";
    cout<<"\033[0;31mPlease provide a valid <param.json> in the current directory."<<endl;
    cout<<"A set of default parameters is leveraged.\033[0m"<<endl;
    cout<< "=============================================================\n";
  } else {
    njson param;
    i >> param;

    // Check param exist or not.
    set<string> paramKeys;
    paramKeys.insert("row");
    paramKeys.insert("column");
    paramKeys.insert("targetFunction");
    paramKeys.insert("kernel");
    paramKeys.insert("targetNested");
    paramKeys.insert("targetLoopsID");
    paramKeys.insert("isTrimmedDemo");
    paramKeys.insert("doCGRAMapping");
    paramKeys.insert("isStaticElasticCGRA");
    paramKeys.insert("ctrlMemConstraint");
    paramKeys.insert("bypassConstraint");
    paramKeys.insert("regConstraint");
    paramKeys.insert("precisionAware");
    paramKeys.insert("vectorizationMode");
    paramKeys.insert("fusionStrategy");
    paramKeys.insert("heuristicMapping");
    paramKeys.insert("parameterizableCGRA");

    try {
      for (auto &k : paramKeys) {
        param.at(k);
      }
    } catch (njson::out_of_range& e) {
      cout<<"Please include related parameter in param.json: "<<e.what()<<endl;
      exit(0);
    }

    (*functionWithLoop)[param["kernel"]] = new list<int>();
    njson loops = param["targetLoopsID"];
    for (int idx = 0; idx < (int)loops.size(); ++idx) {
      (*functionWithLoop)[param["kernel"]]->push_back(loops[idx]);
    }

    // Configuration for customizable CGRA.
    rows                  = param["row"];
    columns               = param["column"];
    targetEntireFunction  = param["targetFunction"];
    targetNested          = param["targetNested"];
    doCGRAMapping         = param["doCGRAMapping"];
    isStaticElasticCGRA   = param["isStaticElasticCGRA"];
    isTrimmedDemo         = param["isTrimmedDemo"];
    ctrlMemConstraint     = param["ctrlMemConstraint"];
    bypassConstraint      = param["bypassConstraint"];
    regConstraint         = param["regConstraint"];
    precisionAware        = param["precisionAware"];
    vectorizationMode     = param["vectorizationMode"];
    heuristicMapping      = param["heuristicMapping"];
    parameterizableCGRA   = param["parameterizableCGRA"];

    if (param.find("incrementalMapping") != param.end()) {
      incrementalMapping = param["incrementalMapping"];
    }
    if (param.find("supportDVFS") != param.end()) {
      supportDVFS = param["supportDVFS"];
    }
    if (param.find("DVFSAwareMapping") != param.end()) {
      DVFSAwareMapping = param["DVFSAwareMapping"];
    }
    if (param.find("DVFSIslandDim") != param.end()) {
      DVFSIslandDim = param["DVFSIslandDim"];
    }
    if (param.find("enablePowerGating") != param.end()) {
      enablePowerGating = param["enablePowerGating"];
    }
    if (param.find("expandableMapping") != param.end()) {
      enableExpandableMapping = param["expandableMapping"];
    }

    // NOTE: original code had a trailing space in the key "vectorFactorForIdiv "
    if (param.find("vectorFactorForIdiv ") != param.end()) {
      vectorFactorForIdiv = param["vectorFactorForIdiv "];
    }
    if (param.find("testingOpcodeOffset") != param.end()) {
      testing_opcode_offset = param["testingOpcodeOffset"];
    }
    if (param.find("multiCycleStrategy") != param.end()) {
      multiCycleStrategy = param["multiCycleStrategy"];
      // Strategy Definition:
      // Exclusive: Multi-cycle ops occupy tiles exclusively.
      // Distributed: Multi-cycle ops split into multiple single-cycle ops.
      // Inclusive: Multi-cycle ops may overlap with other ops on same tile.
      assert(multiCycleStrategy == "exclusive" ||
             multiCycleStrategy == "distributed" ||
             multiCycleStrategy == "inclusive");
    }

    cout<<"Initialize opt latency for DFG nodes: "<<endl;
    for (auto& opt : param["optLatency"].items()) {
      cout<<opt.key()<<" : "<<opt.value()<<endl;
      (*execLatency)[opt.key()] = opt.value();
    }

    njson pipeOpt = param["optPipelined"];
    for (int idx = 0; idx < (int)pipeOpt.size(); ++idx) {
      pipelinedOpt->push_back(pipeOpt[idx]);
    }

    cout<<"Deciding fusion strategy for DFG nodes: "<<endl;
    for (auto& opt : param["fusionStrategy"].items()) {
      fusionStrategy->push_back(opt.value());
    }

    cout<<"Initialize additional functionality on CGRA nodes: "<<endl;
    for (auto& opt : param["additionalFunc"].items()) {
      (*additionalFunc)[opt.key()] = new list<int>();
      cout<<opt.key()<<" : "<<opt.value()<<": ";
      for (int idx = 0; idx < (int)opt.value().size(); ++idx) {
        (*additionalFunc)[opt.key()]->push_back(opt.value()[idx]);
        cout<<opt.value()[idx]<<" ";
      }
      cout<<endl;
    }

    cout<<"Finding fusion pattern for DFG: "<<endl;
    for (auto& opt : param["fusionPattern"].items()) {
      (*fusionPattern)[opt.key()] = new list<string>();
      cout<<opt.key()<<" : "<<opt.value()<<": ";
      for (int idx = 0; idx < (int)opt.value().size(); ++idx) {
        (*fusionPattern)[opt.key()]->push_back(opt.value()[idx]);
        cout<<opt.value()[idx]<<" ";
      }
      cout<<endl;
    }
  }

  // Check existence.
  if (functionWithLoop->find(t_F.getName().str()) == functionWithLoop->end()) {
    cout<<"[function '"<<t_F.getName().str()<<"' is not in our target list]\n";
    return false;
  }
  cout << "==================================\n";
  cout<<"[function '"<<t_F.getName().str()<<"' is one of our targets]\n";

  const bool enableDistributed = (multiCycleStrategy == "distributed");
  const bool enableMultipleOps = (multiCycleStrategy == "inclusive");

  list<llvm::Loop*>* targetLoops = getTargetLoopsImpl(t_F, functionWithLoop, targetNested, LI);

  DFG* dfg = new DFG(t_F, targetLoops, targetEntireFunction, precisionAware,
                    fusionStrategy, execLatency, pipelinedOpt, fusionPattern, supportDVFS,
                    DVFSAwareMapping, vectorFactorForIdiv, enableDistributed);

  if (enableExpandableMapping) {
    dfg->reorderInCriticalFirst();
  }

  CGRA* cgra = new CGRA(rows, columns, vectorizationMode, fusionStrategy,
                       parameterizableCGRA, additionalFunc, supportDVFS,
                       DVFSIslandDim, enableMultipleOps);
  cgra->setRegConstraint(regConstraint);
  cgra->setCtrlMemConstraint(ctrlMemConstraint);
  cgra->setBypassConstraint(bypassConstraint);

  Mapper* mapper = new Mapper(DVFSAwareMapping);

  // Show the count of different opcodes (IRs).
  cout << "==================================\n";
  cout << "[show opcode count]\n";
  dfg->showOpcodeDistribution();

  // Generate the DFG dot file.
  cout << "==================================\n";
  cout << "[generate dot for DFG]\n";
  dfg->generateDot(t_F, isTrimmedDemo);

  // Generate the DFG JSON file.
  cout << "==================================\n";
  cout << "[generate JSON for DFG]\n";
  dfg->generateJSON();

  // Initialize the II.
  int ResMII = mapper->getResMII(dfg, cgra);
  cout << "==================================\n";
  cout << "[ResMII: " << ResMII << "]\n";
  int RecMII = mapper->getRecMII(dfg);
  cout << "==================================\n";
  cout << "[RecMII: " << RecMII << "]\n";

  int II = ResMII;
  if (II < RecMII) II = RecMII;

  if (supportDVFS) {
    dfg->initDVFSLatencyMultiple(II, DVFSIslandDim, cgra->getFUCount());
  }

  if (!doCGRAMapping) {
    cout << "==================================\n";
    return false;
  }
  if (!canMapImpl(cgra, dfg)) {
    cout << "==================================\n";
    cout << "[Mapping Fail]\n";
    return false;
  }

  bool success = false;

  // Heuristic algorithm (hill climbing) to get a valid mapping within a acceptable II.
  if (!isStaticElasticCGRA) {
    cout << "==================================\n";
    using Clock = std::chrono::high_resolution_clock;
    auto t1 = Clock::now();

    if (heuristicMapping) {
      if (incrementalMapping) {
        II = mapper->incrementalMap(cgra, dfg, II);
        cout << "[Incremental]\n";
      } else {
        cout << "[heuristic]\n";
        II = mapper->heuristicMap(cgra, dfg, II, isStaticElasticCGRA);
      }
    } else {
      cout << "[exhaustive]\n";
      II = mapper->exhaustiveMap(cgra, dfg, II, isStaticElasticCGRA);
    }

    auto t2 = Clock::now();
    int elapsedTime =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1000000;
    std::cout <<"Mapping algorithm elapsed time="<<elapsedTime <<"ms"<< '\n';
  }

  // Partially exhaustive search to try to map the DFG onto the static elastic CGRA.
  if (isStaticElasticCGRA && !success) {
    cout << "==================================\n";
    cout << "[exhaustive]\n";
    II = mapper->exhaustiveMap(cgra, dfg, II, isStaticElasticCGRA);
  }

  // Show the mapping and routing results with JSON output.
  if (II == -1) {
    cout << "[fail]\n";
  } else {
    mapper->showSchedule(cgra, dfg, II, isStaticElasticCGRA, parameterizableCGRA);
    cout << "[Mapping Success]\n";
    cout << "==================================\n";
    if (enableExpandableMapping) {
      cout << "[ExpandableII: " << mapper->getExpandableII(dfg, II) << "]\n";
      cout << "==================================\n";
    }
    cout << "[Utilization & DVFS stats]\n";
    mapper->showUtilization(cgra, dfg, II, isStaticElasticCGRA, enablePowerGating);
    cout << "==================================\n";
    mapper->generateJSON(cgra, dfg, II, isStaticElasticCGRA);
    cout << "[Output Json]\n";

    // save mapping results json file for possible incremental mapping
    if (!incrementalMapping) {
      mapper->generateJSON4IncrementalMap(cgra, dfg);
      cout << "[Output Json for Incremental Mapping]\n";
    }
  }

  cout << "=================================="<<endl;

  // Original behavior: does not mutate IR
  return false;
}

/*
 * ----------------------------------------------------------------------
 * NPM pass wrapper
 * ----------------------------------------------------------------------
 */
struct mapperPassNPM : public llvm::PassInfoMixin<mapperPassNPM> {
  llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM) {
    llvm::LoopInfo &LI = FAM.getResult<llvm::LoopAnalysis>(F);

    (void)runMapperImpl(F, LI);

    // Pass doesn't mutate IR.
    return llvm::PreservedAnalyses::all();
  }
};

} // namespace

/*
 * ----------------------------------------------------------------------
 * NPM plugin entry point
 * ----------------------------------------------------------------------
 * Register "mapperPass" as a FUNCTION pipeline element, so invoke with:
 *   opt-21 -load-pass-plugin ./libmapperPass.so -passes='function(mapperPass)' input.ll
 */
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {
    LLVM_PLUGIN_API_VERSION,
    "mapperPass",
    LLVM_VERSION_STRING,
    [](llvm::PassBuilder &PB) {
      PB.registerPipelineParsingCallback(
        [](llvm::StringRef Name, llvm::FunctionPassManager &FPM,
           llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
          if (Name == "mapperPass") {
            FPM.addPass(mapperPassNPM());
            return true;
          }
          return false;
        });
    }
  };
}

/*
 * Add the kernel names of some popular applications.
 * Assume each kernel contains single loop.
 */
void addDefaultKernels(map<string, list<int>*>* t_functionWithLoop) {

  (*t_functionWithLoop)["_Z12ARENA_kerneliii"] = new list<int>();
  (*t_functionWithLoop)["_Z12ARENA_kerneliii"]->push_back(0);
  (*t_functionWithLoop)["_Z4spmviiPiS_S_"] = new list<int>();
  (*t_functionWithLoop)["_Z4spmviiPiS_S_"]->push_back(0);
  (*t_functionWithLoop)["_Z4spmvPiii"] = new list<int>();
  (*t_functionWithLoop)["_Z4spmvPiii"]->push_back(0);
  (*t_functionWithLoop)["adpcm_coder"] = new list<int>();
  (*t_functionWithLoop)["adpcm_coder"]->push_back(0);
  (*t_functionWithLoop)["adpcm_decoder"] = new list<int>();
  (*t_functionWithLoop)["adpcm_decoder"]->push_back(0);
  (*t_functionWithLoop)["kernel_gemm"] = new list<int>();
  (*t_functionWithLoop)["kernel_gemm"]->push_back(0);
  (*t_functionWithLoop)["kernel"] = new list<int>();
  (*t_functionWithLoop)["kernel"]->push_back(0);
  (*t_functionWithLoop)["_Z6kerneli"] = new list<int>();
  (*t_functionWithLoop)["_Z6kerneli"]->push_back(0);
  (*t_functionWithLoop)["_Z6kernelPfPi"] = new list<int>();
  (*t_functionWithLoop)["_Z6kernelPfPi"]->push_back(0);
  (*t_functionWithLoop)["_Z6kernelPfS_"] = new list<int>();
  (*t_functionWithLoop)["_Z6kernelPfS_"]->push_back(0);
  (*t_functionWithLoop)["_Z6kernelPfS_S_"] = new list<int>();
  (*t_functionWithLoop)["_Z6kernelPfS_S_"]->push_back(0);
  (*t_functionWithLoop)["_Z6kerneliPPiS_S_S_"] = new list<int>();
  (*t_functionWithLoop)["_Z6kerneliPPiS_S_S_"]->push_back(0);
  (*t_functionWithLoop)["_Z6kernelPPii"] = new list<int>();
  (*t_functionWithLoop)["_Z6kernelPPii"]->push_back(0);
  (*t_functionWithLoop)["_Z6kernelP7RGBType"] = new list<int>();
  (*t_functionWithLoop)["_Z6kernelP7RGBType"]->push_back(0);
  (*t_functionWithLoop)["_Z6kernelP7RGBTypePi"] = new list<int>();
  (*t_functionWithLoop)["_Z6kernelP7RGBTypePi"]->push_back(0);
  (*t_functionWithLoop)["_Z6kernelP7RGBTypeP4Vect"] = new list<int>();
  (*t_functionWithLoop)["_Z6kernelP7RGBTypeP4Vect"]->push_back(0);
  (*t_functionWithLoop)["fir"] = new list<int>();
  (*t_functionWithLoop)["fir"]->push_back(0);
  (*t_functionWithLoop)["spmv"] = new list<int>();
  (*t_functionWithLoop)["spmv"]->push_back(0);
  // (*functionWithLoop)["fir"].push_back(1);
  (*t_functionWithLoop)["latnrm"] = new list<int>();
  (*t_functionWithLoop)["latnrm"]->push_back(1);
  (*t_functionWithLoop)["fft"] = new list<int>();
  (*t_functionWithLoop)["fft"]->push_back(0);
  (*t_functionWithLoop)["BF_encrypt"] = new list<int>();
  (*t_functionWithLoop)["BF_encrypt"]->push_back(0);
  (*t_functionWithLoop)["susan_smoothing"] = new list<int>();
  (*t_functionWithLoop)["susan_smoothing"]->push_back(0);

  (*t_functionWithLoop)["_Z9LUPSolve0PPdPiS_iS_"] = new list<int>();
  (*t_functionWithLoop)["_Z9LUPSolve0PPdPiS_iS_"]->push_back(0);

  // For LU:
  // init
  (*t_functionWithLoop)["_Z6kernelPPdidPi"] = new list<int>();
  (*t_functionWithLoop)["_Z6kernelPPdidPi"]->push_back(0);

  // solver0 & solver1
  (*t_functionWithLoop)["_Z6kernelPPdPiS_iS_"] = new list<int>();
  (*t_functionWithLoop)["_Z6kernelPPdPiS_iS_"]->push_back(0);

  // determinant
  (*t_functionWithLoop)["_Z6kernelPPdPii"] = new list<int>();
  (*t_functionWithLoop)["_Z6kernelPPdPii"]->push_back(0);

  // invert
  (*t_functionWithLoop)["_Z6kernelPPdPiiS0_"] = new list<int>();
  (*t_functionWithLoop)["_Z6kernelPPdPiiS0_"]->push_back(0);

  (*t_functionWithLoop)["_Z6kernelPiS_i"] = new list<int>();
  (*t_functionWithLoop)["_Z6kernelPiS_i"]->push_back(0);
  (*t_functionWithLoop)["_Z6kernelPfS_f"] = new list<int>();
  (*t_functionWithLoop)["_Z6kernelPfS_f"]->push_back(0);
  (*t_functionWithLoop)["_Z6kernelPiS_"] = new list<int>();
  (*t_functionWithLoop)["_Z6kernelPiS_"]->push_back(0);
  (*t_functionWithLoop)["_Z6kernelPfS_"] = new list<int>();
  (*t_functionWithLoop)["_Z6kernelPfS_"]->push_back(0);
  (*t_functionWithLoop)["_Z6kernelPfS_ff"] = new list<int>();
  (*t_functionWithLoop)["_Z6kernelPfS_ff"]->push_back(0);
  (*t_functionWithLoop)["_Z6kernelPiS_ii"] = new list<int>();
  (*t_functionWithLoop)["_Z6kernelPiS_ii"]->push_back(0);
  (*t_functionWithLoop)["_Z6kernelPfS_if"] = new list<int>();
  (*t_functionWithLoop)["_Z6kernelPfS_if"]->push_back(0);
  (*t_functionWithLoop)["_Z6kernelPiS_S_"] = new list<int>();
  (*t_functionWithLoop)["_Z6kernelPiS_S_"]->push_back(0);
}
