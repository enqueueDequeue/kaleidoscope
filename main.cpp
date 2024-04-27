#include <string>
#include <iostream>
#include <cstdio>
#include <map>
#include <fstream>
#include <vector>
#include <exception>
#include <functional>

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/StandardInstrumentations.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/GVN.h>
#include <llvm/Transforms/Scalar/Reassociate.h>
#include <llvm/Transforms/Scalar/SimplifyCFG.h>
#include <llvm/Support/Error.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/IRCompileLayer.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/ExecutionEngine/Orc/ExecutorProcessControl.h>
#include <llvm/IR/LegacyPassManager.h>

namespace KaleidoScope {

  #define isperiod(x) ('.' == (x))
  #define ispound(x) ('#' == (x))
  #define iseof(x) (EOF == (x))
  #define isterminator(x) (EOF == (x) || '\r' == (x) || '\n' == (x))

  #define log(message, ...) printf(message"\n", ##__VA_ARGS__)

  enum Token {
    // 0-255 represent the input characters themselves
    tok_eof = -1,
    tok_def = -2,
    tok_extern = -3,
    tok_identifier = -4,
    tok_number = -5,
    tok_none = -6,
    tok_if = -7,
    tok_then = -8,
    tok_else = -9,
    tok_for = -10,
    tok_in = -11
  };

  enum Operation {
    op_none = 0,
    op_add,
    op_sub,
    op_mul,
    op_lt
  };

  class JITBag {
    std::unique_ptr<llvm::orc::ExecutionSession> es;
    llvm::Triple target_triple;
    llvm::DataLayout data_layout;
    llvm::orc::RTDyldObjectLinkingLayer object_layer;
    llvm::orc::IRCompileLayer compile_layer;
    llvm::orc::JITDylib& main_jd;

    JITBag(std::unique_ptr<llvm::orc::ExecutionSession> es,
           llvm::Triple target_triple,
           llvm::DataLayout data_layout,
           llvm::orc::JITTargetMachineBuilder jtmb):
           es(std::move(es)),
           target_triple(std::move(target_triple)),
           data_layout(std::move(data_layout)),
           object_layer(*this->es, []() { return std::make_unique<llvm::SectionMemoryManager>(); }),
           compile_layer(*this->es, object_layer, std::make_unique<llvm::orc::ConcurrentIRCompiler>(std::move(jtmb))),
           main_jd(this->es->createBareJITDylib("<main>")) {

      main_jd.addGenerator(llvm::cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(data_layout.getGlobalPrefix())));
    }

    public:
    ~JITBag() {
      if (llvm::Error err = es->endSession()) {
        es->reportError(std::move(err));
      }
    }

    static JITBag create() {
      llvm::InitializeNativeTarget();

      std::unique_ptr<llvm::orc::SelfExecutorProcessControl> epc = llvm::cantFail(llvm::orc::SelfExecutorProcessControl::Create());
      std::unique_ptr<llvm::orc::ExecutionSession> es = std::make_unique<llvm::orc::ExecutionSession>(std::move(epc));

      llvm::Triple target_triple = es->getExecutorProcessControl().getTargetTriple();
      llvm::orc::JITTargetMachineBuilder jtmb(target_triple);
      llvm::DataLayout data_layout = llvm::cantFail(jtmb.getDefaultDataLayoutForTarget());

      return JITBag(std::move(es), std::move(target_triple), std::move(data_layout), std::move(jtmb));
    }

    const llvm::Triple& get_target_triple() const {
      return target_triple;
    }

    const llvm::DataLayout& get_data_layout() const {
      return data_layout;
    }
  };

  class IRBag {
    llvm::LLVMContext context;
    llvm::Module module;
    llvm::IRBuilder<> builder;
    std::map<std::string, llvm::Value*> var_map;

    public:
    llvm::FunctionPassManager fpm;
    llvm::LoopAnalysisManager lam;
    llvm::FunctionAnalysisManager fam;
    llvm::CGSCCAnalysisManager cgam;
    llvm::ModuleAnalysisManager mam;
    llvm::PassInstrumentationCallbacks pic;
    llvm::StandardInstrumentations si;
    llvm::PassBuilder pb;

    IRBag(JITBag& jit_bag): context(), module("The jit", context), builder(context), si(context, true) {
      module.setDataLayout(jit_bag.get_data_layout());

      si.registerCallbacks(pic, &mam);

      fpm.addPass(llvm::InstCombinePass());
      fpm.addPass(llvm::ReassociatePass());
      fpm.addPass(llvm::GVNPass());
      fpm.addPass(llvm::SimplifyCFGPass());

      pb.registerModuleAnalyses(mam);
      pb.registerFunctionAnalyses(fam);
      pb.crossRegisterProxies(lam, fam, cgam, mam);
    }

    llvm::LLVMContext& get_context() {
      return context;
    }

    llvm::Module& get_module() {
      return module;
    }

    llvm::IRBuilder<>& get_builder() {
      return builder;
    }

    void clear_vars() {
      var_map.clear();
    }

    llvm::Value* get_var(const std::string& name) {
      // log("Getting name: %s", name.c_str());
      return var_map[name];
    }

    void set_var(const std::string& name, llvm::Value* value) {
      // log("Setting name: %s", name.c_str());
      var_map[name] = value;
    }

    void erase_var(const std::string& name) {
      var_map.erase(name);
    }
  };

  class AST {
    public:
    virtual ~AST() = default;
    virtual llvm::Value* to_llvm_ir(IRBag& ir_bag) = 0;
  };

  class ExprAST: public AST {
    public:
    ExprAST() {}
    virtual ~ExprAST() = default;
    virtual std::string dump() = 0;
  };

  class NumberExprAST: public ExprAST {
    double value;

    public:
    NumberExprAST(double value): value(value) {}

    llvm::Value* to_llvm_ir(IRBag& ir_bag) override {
      return llvm::ConstantFP::get(ir_bag.get_context(), llvm::APFloat(value));
    }

    std::string dump() override {
      return "NumberExprAST: " + std::to_string(value);
    }
  };

  class VarExprAST: public ExprAST {
    std::string name;

    public:
    VarExprAST(std::string name): name(name) {}

    llvm::Value* to_llvm_ir(IRBag& ir_bag) override {
      if (llvm::Value* v = ir_bag.get_var(name)) {
        return v;
      }

      log("ERROR: Cannot find the variable: %s", name.c_str());
      return nullptr;
    }

    std::string dump() override {
      return "VarExprAST: " + name;
    }
  };

  class BinaryExprAST: public ExprAST {
    Operation op;
    std::unique_ptr<ExprAST> lhs;
    std::unique_ptr<ExprAST> rhs;

    public:
    BinaryExprAST(Operation op,
                  std::unique_ptr<ExprAST> lhs,
                  std::unique_ptr<ExprAST> rhs):
                  op(op),
                  lhs(std::move(lhs)),
                  rhs(std::move(rhs)) {}

    llvm::Value* to_llvm_ir(IRBag& ir_bag) override {
      llvm::Value* lhs = this->lhs->to_llvm_ir(ir_bag);
      llvm::Value* rhs = this->rhs->to_llvm_ir(ir_bag);

      switch (op) {
        case op_add: {
          return ir_bag.get_builder().CreateFAdd(lhs, rhs, "add_tmp");
        }
        case op_sub: {
          return ir_bag.get_builder().CreateFSub(lhs, rhs, "sub_tmp");
        }
        case op_mul: {
          return ir_bag.get_builder().CreateFMul(lhs, rhs, "mul_tmp");
        }
        case op_lt: {
          llvm::Value* cmp = ir_bag.get_builder().CreateFCmpULT(lhs, rhs, "cmp_tmp");
          return ir_bag.get_builder().CreateUIToFP(cmp, llvm::Type::getDoubleTy(ir_bag.get_context()), "bool_tmp");
        }
        default: {
          log("Error: Cannot convert op to LLVM IR");
          return nullptr;
        }
      }
    }

    std::string dump() override {
      return "" + std::to_string(op) + lhs->dump() + rhs->dump();
    }
  };

  class CallExprAST: public ExprAST {
    std::string callee;
    std::vector<std::unique_ptr<ExprAST>> args;

    public:
    CallExprAST(std::string callee,
                std::vector<std::unique_ptr<ExprAST>> args):
                callee(callee),
                args(std::move(args)) {}

    llvm::Value* to_llvm_ir(IRBag& ir_bag) override {
      std::vector<llvm::Value*> args;
      llvm::Function* func = ir_bag.get_module().getFunction(callee);

      if (func->arg_size() != this->args.size()) {
        log("Error: Argument mismatch: func: %zu, args: %zu", func->arg_size(), this->args.size());
        return nullptr;
      }

      for (int i = 0; i < this->args.size(); i++) {
        args.push_back(this->args[i]->to_llvm_ir(ir_bag));
      }

      return ir_bag.get_builder().CreateCall(func, args, "call_tmp");
    }

    std::string dump() override {
      std::string info = "";

      info += callee;

      for (auto &arg: args) {
        info += (" " + arg->dump());
      }

      return info;
    }
  };

  class FunctionAST: public AST {
    std::string name;
    std::vector<std::string> args;
    std::unique_ptr<ExprAST> body;

    public:
    FunctionAST(std::string name,
                std::vector<std::string> args,
                std::unique_ptr<ExprAST> body):
                name(name),
                args(args),
                body(std::move(body)) {}

    llvm::Value* to_llvm_ir(IRBag& ir_bag) override {
      std::vector<llvm::Type*> arg_types(args.size(), llvm::Type::getDoubleTy(ir_bag.get_context()));
      llvm::FunctionType* func_type = llvm::FunctionType::get(llvm::Type::getDoubleTy(ir_bag.get_context()), arg_types, false);

      llvm::Function* func = llvm::Function::Create(func_type, llvm::Function::ExternalLinkage, name, ir_bag.get_module());

      {
        unsigned idx = 0;

        for (auto& arg: func->args()) {
          arg.setName(args[idx]);
          idx += 1;
        }
      }

      llvm::BasicBlock *bb = llvm::BasicBlock::Create(ir_bag.get_context(), name, func);

      ir_bag.get_builder().SetInsertPoint(bb);

      ir_bag.clear_vars();

      for (auto &arg : func->args()) {
        // log("dbg: setting arg: %s", std::string(arg.getName()).c_str());
        ir_bag.set_var(std::string(arg.getName()), &arg);
      }

      if (llvm::Value* ret_val = body->to_llvm_ir(ir_bag)) {
        ir_bag.get_builder().CreateRet(ret_val);

        llvm::verifyFunction(*func);

        // ir_bag.fpm.run(*func, ir_bag.fam);

        return func;
      }

      func->eraseFromParent();
      return nullptr;
    }
  };

  class ExternAST: public AST {
    std::string name;
    std::vector<std::string> args;

    public:
    ExternAST(std::string name,
              std::vector<std::string> args):
              name(name),
              args(args) {}

    llvm::Value* to_llvm_ir(IRBag& ir_bag) override {
      std::vector<llvm::Type*> arg_types(this->args.size(), llvm::Type::getDoubleTy(ir_bag.get_context()));
      llvm::FunctionType* func_type = llvm::FunctionType::get(llvm::Type::getDoubleTy(ir_bag.get_context()), arg_types, false);

      llvm::Function* func = llvm::Function::Create(func_type, llvm::Function::ExternalLinkage, name, ir_bag.get_module());

      {
        unsigned idx = 0;

        for (auto& arg: func->args()) {
          arg.setName(args[idx]);
          idx += 1;
        }
      }

      return func;
    }
  };

  class IfExprAST: public ExprAST {
    std::unique_ptr<ExprAST> cond;
    std::unique_ptr<ExprAST> then;
    std::unique_ptr<ExprAST> otherwise;

    public:
    IfExprAST(std::unique_ptr<ExprAST> cond,
              std::unique_ptr<ExprAST> then,
              std::unique_ptr<ExprAST> otherwise):
              cond(std::move(cond)),
              then(std::move(then)),
              otherwise(std::move(otherwise)) {}

    llvm::Value* to_llvm_ir(IRBag& ir_bag) override {
      llvm::Value* cond_value = cond->to_llvm_ir(ir_bag);

      // 1. create the condition
      // FCmpONE = float compare ordered(?) not equals
      llvm::Value* cmp_value = ir_bag.get_builder().CreateFCmpONE(cond_value,
                                                    llvm::ConstantFP::get(ir_bag.get_context(), llvm::APFloat(0.0)),
                                                    "cond");

      llvm::Function* parent_function = ir_bag.get_builder().GetInsertBlock()->getParent();

      llvm::BasicBlock* then_block = llvm::BasicBlock::Create(ir_bag.get_context(), "if.then");
      llvm::BasicBlock* otherwise_block = llvm::BasicBlock::Create(ir_bag.get_context(), "if.otherwise");
      llvm::BasicBlock* merge_block = llvm::BasicBlock::Create(ir_bag.get_context(), "if.finally");

      // 2. create the branch
      ir_bag.get_builder().CreateCondBr(cmp_value, then_block, otherwise_block);

      // 3. create the then block
      parent_function->insert(parent_function->end(), then_block);
      ir_bag.get_builder().SetInsertPoint(then_block);

      // this will be inserted automatically
      // this will also be used during the disambiguation in the phi node
      llvm::Value* then_value = then->to_llvm_ir(ir_bag);

      // the jmp instruction after a branch
      ir_bag.get_builder().CreateBr(merge_block);

      // During the code generation, the basic block pointers might change
      // So, get the new pointer again
      then_block = ir_bag.get_builder().GetInsertBlock();

      // 4. create the else block
      parent_function->insert(parent_function->end(), otherwise_block);
      ir_bag.get_builder().SetInsertPoint(otherwise_block);

      llvm::Value* otherwise_value = otherwise->to_llvm_ir(ir_bag);

      ir_bag.get_builder().CreateBr(merge_block);

      otherwise_block = ir_bag.get_builder().GetInsertBlock();

      // 5. create the merge block
      parent_function->insert(parent_function->end(), merge_block);

      ir_bag.get_builder().SetInsertPoint(merge_block);

      llvm::PHINode* phi_node = ir_bag.get_builder().CreatePHI(llvm::Type::getDoubleTy(ir_bag.get_context()), 2);

      phi_node->addIncoming(then_value, then_block);
      phi_node->addIncoming(otherwise_value, otherwise_block);

      return phi_node;
    }

    std::string dump() override {
      std::string info;

      info += "if (" + cond->dump() + "):";
      info += then->dump();
      info += "else:";
      info += otherwise->dump();

      return info;
    }
  };

  class ForExprAST: public ExprAST {
    std::string name;
    std::unique_ptr<ExprAST> start;
    std::unique_ptr<ExprAST> end;
    std::unique_ptr<ExprAST> step;
    std::unique_ptr<ExprAST> body;

    public:
    ForExprAST(std::string name,
               std::unique_ptr<ExprAST> start,
               std::unique_ptr<ExprAST> end,
               std::unique_ptr<ExprAST> step,
               std::unique_ptr<ExprAST> body):
               name(name),
               start(std::move(start)),
               end(std::move(end)),
               step(std::move(step)),
               body(std::move(body)) {}

    llvm::Value* to_llvm_ir(IRBag& ir_bag) override {
      llvm::Function* parent_function = ir_bag.get_builder().GetInsertBlock()->getParent();

      llvm::BasicBlock* header_block = ir_bag.get_builder().GetInsertBlock();
      llvm::BasicBlock* cond_block = llvm::BasicBlock::Create(ir_bag.get_context(), "for.cond");
      llvm::BasicBlock* body_block = llvm::BasicBlock::Create(ir_bag.get_context(), "for.body");
      llvm::BasicBlock* finally_block = llvm::BasicBlock::Create(ir_bag.get_context(), "for.finally");

      // header block begin
      llvm::Value* start_value = start->to_llvm_ir(ir_bag);
      ir_bag.get_builder().CreateBr(cond_block);

      header_block = ir_bag.get_builder().GetInsertBlock();

      // cond block begin
      parent_function->insert(parent_function->end(), cond_block);
      ir_bag.get_builder().SetInsertPoint(cond_block);

      llvm::PHINode* loop_idx = ir_bag.get_builder().CreatePHI(llvm::Type::getDoubleTy(ir_bag.get_context()), 2);

      llvm::Value* old_value = ir_bag.get_var(name);
      ir_bag.set_var(name, loop_idx);

      // this is actually the condition
      // 0.0 means false, anything else is true
      llvm::Value* end_value = end->to_llvm_ir(ir_bag);
      llvm::Value* cond_value = ir_bag.get_builder().CreateFCmpONE(end_value,
                                                                   llvm::ConstantFP::get(ir_bag.get_context(), llvm::APFloat(0.0)),
                                                                   "cond");

      ir_bag.get_builder().CreateCondBr(cond_value, body_block, finally_block);

      // body block begin
      parent_function->insert(parent_function->end(), body_block);
      ir_bag.get_builder().SetInsertPoint(body_block);

      llvm::Value* body_value = body->to_llvm_ir(ir_bag);

      if (!body_value) {
        log("Error: Cannot generate the body");
        return nullptr;
      }

      llvm::Value* step_value = step->to_llvm_ir(ir_bag);

      llvm::Value* next_idx = ir_bag.get_builder().CreateFAdd(loop_idx, step_value, "next_idx");

      ir_bag.get_builder().CreateBr(cond_block);

      body_block = ir_bag.get_builder().GetInsertBlock();

      loop_idx->addIncoming(start_value, header_block);
      loop_idx->addIncoming(next_idx, body_block);

      // exit block begin
      // finally block doesn't have any instrctions
      // this is for the next parsers to insert into
      parent_function->insert(parent_function->end(), finally_block);

      ir_bag.get_builder().SetInsertPoint(finally_block);

      if (old_value) {
        ir_bag.set_var(name, old_value);
      } else {
        ir_bag.erase_var(name);
      }

      return llvm::Constant::getNullValue(llvm::Type::getDoubleTy(ir_bag.get_context()));
    }

    std::string dump() override {
      std::string info;

      info += "for(" + name + "=";
      info += start->dump();
      info += ", ";
      info += end->dump();
      info += ", ";
      info += step->dump();
      info += " in ";
      info += body->dump();
      info += ";";

      return info;
    }
  };

  class Compiler {
    std::function<char()> read;
    std::string input;
    int last_char;
    int current_token;

    public:
    Compiler(std::function<char()> read): read(read),
                                            input(""),
                                            last_char(' '),
                                            current_token(tok_none) {}

    ~Compiler() {}

    int get_next_token() {
      return current_token = get_token();
    }

    int get_token() {
      while (isspace(last_char)) {
        last_char = read();
      }

      if (isalpha(last_char)) {
        input = "";

        do {
          input += last_char;
          last_char = read();
        } while (isalnum(last_char));

        if (input == "def") {
          return tok_def;
        } else if (input == "extern") {
          return tok_extern;
        } else if (input == "if") {
          return tok_if;
        } else if (input == "then") {
          return tok_then;
        } else if (input == "else") {
          return tok_else;
        } else if (input == "for") {
          return tok_for;
        } else if (input == "in") {
          return tok_in;
        } else {
          return tok_identifier;
        }
      }

      // Assuming numbers cannot start with .
      // This is a pedantic language
      if (isdigit(last_char)) {
        input = "";

        do {
          input += last_char;
          last_char = read();
        } while (isdigit(last_char) || isperiod(last_char));

        return tok_number;
      }

      if (ispound(last_char)) {
        do {
          last_char = read();
        } while (!isterminator(last_char));

        return tok_none;
      }

      if (iseof(last_char)) {
        return tok_eof;
      }

      int this_char = last_char;
      last_char = read();
      return this_char;
    }

    std::unique_ptr<ExprAST> parse_if() {
      get_next_token();

      std::unique_ptr<ExprAST> cond = parse_expression();

      if (!cond) {
        log("Error: cannot parse condition");
        return nullptr;
      }

      if (current_token != tok_then) {
        log("Error: Expecting then token, found: %d, input: %s", current_token, input.c_str());
        return nullptr;
      }

      get_next_token();

      std::unique_ptr<ExprAST> then = parse_expression();

      if (current_token != tok_else) {
        log("Error: Expecting else token");
        return nullptr;
      }

      get_next_token();

      std::unique_ptr<ExprAST> otherwise = parse_expression();

      return std::make_unique<IfExprAST>(std::move(cond), std::move(then), std::move(otherwise));
    }

    std::unique_ptr<ExprAST> parse_for() {
      get_next_token();

      if (current_token != tok_identifier) {
        log("Error: expecting an identifier");
        return nullptr;
      }

      std::string loop_idx = input;
      get_next_token();

      if (current_token != '=') {
        log("Error: Expecting an =");
        return nullptr;
      }
      get_next_token();

      std::unique_ptr<ExprAST> start = parse_expression();
      if (!start) {
        log("Error: Cannot parse for start");
        return nullptr;
      }
      if (current_token != ',') {
        log("Error: Expecting a ,");
        return nullptr;
      }
      get_next_token();

      std::unique_ptr<ExprAST> end = parse_expression();
      if (!end) {
        log("Error: Cannot parse for end");
        return nullptr;
      }
      if (current_token != ',') {
        log("Error: Expecting a ,");
        return nullptr;
      }
      get_next_token();

      std::unique_ptr<ExprAST> step = parse_expression();

      if (!step) {
        log("Error: Cannot parse for end");
        return nullptr;
      }

      if (current_token != tok_in) {
        log("Error: Expecting an in");
        return nullptr;
      }

      get_next_token();

      std::unique_ptr<ExprAST> body = parse_expression();

      if (!body) {
        log("Error: Cannot parse for body");
        return nullptr;
      }

      return std::make_unique<ForExprAST>(loop_idx,
                                          std::move(start),
                                          std::move(end),
                                          std::move(step),
                                          std::move(body));
    }

    std::unique_ptr<ExprAST> parse_number() {
      auto expr = std::make_unique<NumberExprAST>(std::stod(input));

      // Get the token for the next round of parsing
      // This also indicates that we successfully parsed
      // the current token
      get_next_token();

      return expr;
    }

    std::unique_ptr<ExprAST> parse_identifier() {
      std::string name = input;

      // two possible situations:
      // 1) just a variable
      // 2) a function call

      get_next_token();

      if (current_token == '(') {
        // function call
        get_next_token();

        std::vector<std::unique_ptr<ExprAST>> args;

        while (true) {
          if (std::unique_ptr<ExprAST> expr = parse_expression()) {
            // log("dbg: parsed arg: %s", expr->dump().c_str());

            args.push_back(std::move(expr));
          } else {
            log("Error: Cannot parse argument expression");
            return nullptr;
          }

          if (current_token == ')') {
            break;
          }

          if (current_token != ',') {
            log("Error: Expecting , between consecutive argument, found: %d (%c)", current_token, (char) current_token);
            return nullptr;
          }

          get_next_token();
        }

        get_next_token();

        return std::make_unique<CallExprAST>(name, std::move(args));
      } else {
        // variable
        return std::make_unique<VarExprAST>(name);
      }
    }

    std::unique_ptr<FunctionAST> parse_definiton() {
      get_next_token();

      std::string name = input;

      get_next_token();

      if (current_token != '(') {
        log("Error: Expecting (");
        return nullptr;
      }

      std::vector<std::string> args;

      while (tok_identifier == get_next_token()) {
        args.push_back(input);
      }

      if (current_token != ')') {
        log("Error: Expecting ), found: %d (%c)", current_token, (char) current_token);
        return nullptr;
      }

      get_next_token();

      std::unique_ptr<ExprAST> body = parse_expression();

      return std::make_unique<FunctionAST>(name, args, std::move(body));
    }

    std::unique_ptr<ExternAST> parse_extern() {
      get_next_token();

      std::string name = input;

      get_next_token();

      if (current_token != '(') {
        log("Error: Expecting (");
        return nullptr;
      }

      std::vector<std::string> args;

      while (tok_identifier == get_next_token()) {
        args.push_back(input);
      }

      if (current_token != ')') {
        log("Error: Expecting )");
        return nullptr;
      }

      get_next_token();

      return std::make_unique<ExternAST>(name, args);
    }

    std::unique_ptr<ExprAST> parse_parenthesis() {
      get_next_token();

      auto expr = parse_expression();

      if (!expr) {
        log("Error: Cannot parse parenthesis expression");
        return nullptr;
      }

      if (current_token != ')') {
        log("Error: expecting )");
        return nullptr;
      }

      get_next_token();
      return expr;
    }

    std::unique_ptr<ExprAST> parse_primary() {
      switch (current_token) {
        case tok_identifier: {
          return parse_identifier();
        }
        case tok_number: {
          return parse_number();
        }
        case tok_if: {
          return parse_if();
        }
        case tok_for: {
          return parse_for();
        }
        case '(': {
          return parse_parenthesis();
        }
        default: {
          log("illegal token: %d (%c)", current_token, (char) current_token);
          return nullptr;
        }
      }
    }

    int get_token_precedence() {
      // higher the number higher the priority
      // say, this token is a new def
      // then this will invoke the lhs in
      // parse_bin_op to return
      if (!isascii(current_token)) {
        return -1;
      }

      int token_precedence;

      switch (current_token) {
        case '<': {
          token_precedence = 10;
          break;
        }
        case '+': {
          token_precedence = 20;
          break;
        }
        case '-': {
          token_precedence = 20;
          break;
        }
        case '*': {
          token_precedence = 40;
          break;
        }
        default: {
          token_precedence = 0;
          break;
        }
      }

      if (token_precedence <= 0) {
        return -1;
      }

      return token_precedence;
    }

    Operation convert_to_op(int token) {
      switch (token) {
        case '<': return op_lt;
        case '+': return op_add;
        case '-': return op_sub;
        case '*': return op_mul;
        default: return op_none;
      }
    }

    std::unique_ptr<ExprAST> parse_bin_op(int precedence,
                                          std::unique_ptr<ExprAST> lhs) {

      while (true) {
        int token_precedence = get_token_precedence();

        // magic
        if (token_precedence < precedence) {
          return lhs;
        }

        Operation bin_op = convert_to_op(current_token);

        get_next_token();

        auto rhs = parse_primary();

        if (!rhs) {
          log("Error: Cannot parse RHS of the binary op");
          return nullptr;
        }

        int next_precedence = get_token_precedence();

        if (token_precedence < next_precedence) {
          rhs = parse_bin_op(precedence + 1, std::move(rhs));

          if (!rhs) {
            log("Error: recursive parse failed");
            return nullptr;
          }
        }

        lhs = std::make_unique<BinaryExprAST>(bin_op, std::move(lhs), std::move(rhs));
      }
    }

    std::unique_ptr<ExprAST> parse_expression() {
      if (auto lhs = parse_primary()) {
        // try to parse this as a binary expression
        // if we can't then the function itself will
        // return the lhs

        return parse_bin_op(0, std::move(lhs));
      }

      return nullptr;
    }

    void compile(IRBag& ir_bag) {
      int tok;

      get_next_token();

      while (true) {
        switch (current_token) {
          case tok_eof: {
            return;
          }
          case ';':
          case tok_none: {
            get_next_token();
            break;
          }
          case tok_def: {
            auto fn_ast = parse_definiton();
            auto fn_ir = fn_ast->to_llvm_ir(ir_bag);

            fn_ir->print(llvm::errs());
            fprintf(stderr, "\n");
            break;
          }
          case tok_extern: {
            auto fn_ast = parse_extern();
            auto fn_ir = fn_ast->to_llvm_ir(ir_bag);

            fn_ir->print(llvm::errs());
            fprintf(stderr, "\n");
            break;
          }
          default: {
            log("Top Level functions not supported yet");

            // // parse for an anonymous expression
            // if (auto expr = parse_expression()) {
            //   auto fn_ast = std::make_unique<FunctionAST>("__anon_expr", std::vector<std::string>(), std::move(expr));

            //   auto fn_ir = fn_ast->to_llvm_ir(ir_bag);

            //   fn_ir->print(llvm::errs());
            //   fprintf(stderr, "\n");
            // } else {
            //   log("Error: Cannot parse anonymous expression");
            // }

            break;
          }
        }
      }
    }
  };
}

int main(int argc, char **argv) {
  if (3 != argc) {
    log("Error: input & output file needs to be passed");
    return -1;
  }

  const char* src = argv[1];
  const char* dst = argv[2];
  std::ifstream input(src);

  std::error_code output_ec;
  llvm::raw_fd_ostream output(dst, output_ec, llvm::sys::fs::OF_None);

  if (!input) {
    log("Error: Cannot open the input file");
    return -2;
  }

  if (output_ec) {
    log("Error: Cannot open the output file");
    return -3;
  }

  KaleidoScope::JITBag jit_bag = KaleidoScope::JITBag::create();

  KaleidoScope::IRBag ir_bag(jit_bag);

  KaleidoScope::Compiler c([&input] { char c; if (input.get(c)) { return c; } else { return (char) EOF; } });

  c.compile(ir_bag);

  llvm::TargetOptions options;
  llvm::Triple target_triple = jit_bag.get_target_triple();

  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();

  std::string error;
  const llvm::Target* target = llvm::TargetRegistry::lookupTarget(target_triple.getArchName(), target_triple, error);
  llvm::TargetMachine* target_machine = target->createTargetMachine(target_triple.getTriple(), "generic", "", options, llvm::Reloc::PIC_);

  ir_bag.get_module().setDataLayout(target_machine->createDataLayout());

  llvm::legacy::PassManager pass;

  if (target_machine->addPassesToEmitFile(pass, output, nullptr, llvm::CodeGenFileType::CGFT_ObjectFile)) {
    log("Error: Cannot output to file");
    return -4;
  }

  pass.run(ir_bag.get_module());
  output.flush();

  return 0;
}
