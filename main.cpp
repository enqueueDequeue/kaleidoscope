#include <string>
#include <cstdio>
#include <vector>
#include <exception>
#include <functional>

namespace KaleidoScope {

  #define isperiod(x) ('.' == (x))
  #define ispound(x) ('#' == (x))
  #define iseof(x) (EOF == (x))
  #define isterminator(x) (EOF == (x) || '\r' == (x) || '\n' == (x))

  enum Token {
    // 0-255 represent the input characters themselves
    tok_eof = -1,
    tok_def = -2,
    tok_extern = -3,
    tok_identifier = -4,
    tok_number = -5,
    tok_none = -6
  };

  enum Operation {
    op_none = 0,
    op_add,
    op_sub,
    op_mul,
    op_lt
  };

  class ExprAST {
    public:
    ExprAST() {}
    virtual ~ExprAST() = default;
  };

  class NumberExprAST: public ExprAST {
    double value;

    public:
    NumberExprAST(double value): value(value) {}
  };

  class VarExprAST: public ExprAST {
    std::string name;

    public:
    VarExprAST(std::string name): name(name) {}
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
  };

  class CallExprAST: public ExprAST {
    std::string callee;
    std::vector<std::unique_ptr<ExprAST>> args;

    public:
    CallExprAST(std::string callee,
                std::vector<std::unique_ptr<ExprAST>> args):
                callee(callee),
                args(std::move(args)) {}
  };

  class FunctionAST {
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
  };

  class ExternAST {
    std::string name;
    std::vector<std::string> args;

    public:
    ExternAST(std::string name,
              std::vector<std::string> args):
              name(name),
              args(args) {}
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

    void log(const std::string &message) {
      printf("%s", message.c_str());
    }

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

        while (current_token != ')') {
          if (std::unique_ptr<ExprAST> expr = parse_expression()) {
            args.push_back(std::move(expr));
          } else {
            log("Error: Cannot parse argument expression");
            return nullptr;
          }

          if (current_token != ',') {
            log("Error: Expecting , between consecutive argument");
            return nullptr;
          }

          get_next_token();
        }

        get_next_token();

        return std::make_unique<CallExprAST>(name, std::move(args));
      } else {
        // variable
        get_next_token();
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
        log("Error: Expecting )");
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

    std::unique_ptr<ExprAST> parse_primary() {
      switch (current_token) {
        case tok_identifier: {
          return parse_identifier();
        }
        case tok_number: {
          return parse_number();
        }
        default: {
          log("illegal token");
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

    void compile() {
      int tok;

      while (tok_eof != (tok = get_token())) {
        printf("%d\n", tok);
      }
    }
  };
}

int main() {
  KaleidoScope::Compiler c([] { return getchar(); });

  c.compile();
}
