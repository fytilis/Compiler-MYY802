# Greek++ Compiler

A complete, multi-stage compiler for the educational programming language **Greek++**. The compiler is written entirely in Python and translates Greek++ source code into assembly for the **RISC-V** architecture.

This project was developed as part of the "MYY802 Compilers" course at the Department of Computer Science & Engineering, University of Ioannina.

## 📖 Greek++ Language Features

Greek++ is a simple, procedural language designed for educational purposes, featuring syntax based on the Greek language. It supports:

*   **Variable Declarations**: Integer variables.
*   **Assignment Statements**: Using the `:=` operator.
*   **Control Flow Structures**:
    *   `if-then-else` for conditional logic.
    *   `while`, `repeat-until`, and `for` for loops.
*   **Subprograms**:
    *   `function` (returns a value).
    *   `procedure` (does not return a value).
*   **Parameter Passing Mechanisms**:
    *   **Call by Value** (`in` parameter).
    *   **Call by Reference** (`out` parameter), using the special character `%` during the call.
*   **Input/Output**: `read` and `write` commands.
*   **Recursive Calls** of subprograms.
*   **Comments** enclosed in `{` and `}`, which support nesting.

## ⚙️ Compiler Architecture

The compiler is implemented as a classic four-phase pipeline:

1.  **Lexical Analyzer (Lexer)**
    *   Scans the source code and converts it into a sequence of tokens.
    *   Recognizes keywords, identifiers (variables), numbers, operators, and comments.
    *   Manages and reports lexical errors (e.g., invalid characters, unclosed comments).

2.  **Syntactic & Semantic Analyzer (Parser)**
    *   Implemented using the **Recursive Descent** method.
    *   Verifies that the token sequence adheres to the rules of the Greek++ grammar.
    *   Constructs an **Abstract Syntax Tree (AST)** that represents the hierarchical structure of the program.
    *   In conjunction with the Symbol Table, it performs semantic checks, such as checking for duplicate variable declarations or the use of undeclared identifiers.

3.  **Intermediate Code Generator**
    *   Traverses the AST and generates intermediate code in the form of **quadruples**.
    *   Uses the **backpatching** technique (with `truelist` and `falselist`) to manage jumps in control flow structures.
    *   Handles expressions based on operator precedence (precedence climbing).
    *   Creates temporary variables (`t@1`, `t@2`, ...) to store intermediate results.

4.  **Final Code Generator**
    *   Translates the intermediate code quadruples into final **RISC-V assembly** code.
    *   Implements the stack frame management protocol to support local variables and subprograms.
    *   Includes the critical routines `loadvr` (load value into a register), `storerv` (store value from a register), and `gnvlcode` (access non-local variables via the static chain, or access links).

## 🚀 How to Run

### Prerequisites

You only need **Python 3** installed on your system.

### Usage

1.  Create a source code file with a `.gr` extension (e.g., `example.gr`). See the `greek.gr` file for an example.

2.  Run the compiler from the command line, providing your source file as an argument:
    ```bash
    python greekplusplus.py <your_file>.gr
    ```
    For example:
    ```bash
    python greekplusplus.py example.gr
    ```

3.  The compiler will generate three output files in the same directory:
    *   **`example.int`**: The intermediate code in quadruple format.
    *   **`example.sym`**: The contents of the Symbol Table, useful for debugging.
    *   **`example.asm`**: The final executable code in RISC-V assembly.

4.  You can execute the `.asm` file using a RISC-V simulator, such as [RARS](https://github.com/TheThirdOne/rars).

## ✍️ Authors

*   **Theofanis Tombolis** (cs04855)
*   **Athanasios Fytilis** (cs05831)
