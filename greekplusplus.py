# THEOFANIS TOMPOLIS 4855 username:cs04855
# ATHANASIOS FYTILIS 5381 username:cs05831
import pprint
import sys

# --- Constants ---
KEYWORDS = {
    "πρόγραμμα", "δήλωση", "εάν", "τότε", "αλλιώς", "εάν_τέλος", "επανάλαβε",
    "μέχρι", "όσο", "όσο_τέλος", "για", "έως", "με_βήμα", "για_τέλος",
    "διάβασε", "γράψε", "συνάρτηση", "διαδικασία", "διαπροσωπεία",
    "είσοδος", "έξοδος", "αρχή_συνάρτησης", "τέλος_συνάρτησης",
    "αρχή_διαδικασίας", "τέλος_διαδικασίας", "αρχή_προγράμματος",
    "τέλος_προγράμματος", "ή", "και", "εκτέλεσε", "όχι"
}

OPERATORS = {'+', '-', '*', '/', '=', '==', '<', '>', '<=', '>=', '<>', ':=', '<-'}
RELATIONAL_OPERATORS = {'=', '==', '<', '>', '<=', '>=', '<>', '<-'}
LOGICAL_OPERATORS = {'και', 'ή', 'όχι'}
ARITHMETIC_OPERATORS = {'+', '-', '*', '/'}
ASSIGNMENT = ':='
PUNCTUATION = {';', ',', '(', ')', '{', '}', '[', ']'}
PARAMETER_PASS = '%'

# Operator precedence for ICG expression evaluation
OP_PRECEDENCE = {
    '(': 0, ')': 0, '[': 0, ']': 0,  # Grouping 
    'ή': 1,  # Logical OR
    'και': 2,  # Logical AND
    'όχι': 3,  # Logical NOT 
    '=': 4, '==': 4, '<': 4, '>': 4, '<=': 4, '>=': 4, '<>': 4, '<-': 4,  # Relational
    '+': 5, '-': 5,  # Add/Subtract
    '*': 6, '/': 6,  # Multiply/Divide
    'UPLUS': 7, 'UMINUS': 7  # Unary +/- 
}

# --- Lexer ---
def lexer(code):
    """
    Tokenizes the input GreekPlusPlus source code.
    Returns:
        list: A list of tokens, where each token is a tuple:
              (TYPE, VALUE, LINE_NUMBER) or ('MISMATCH', ERROR_MESSAGE, LINE_NUMBER).
    """
    i = 0
    tokens = []
    length = len(code)
    line = 1
    while i < length:
        char = code[i]
        # --- Whitespace and Comments ---
        if char in ' \t':
            i += 1
            continue
        if char == '\n':
            line += 1
            i += 1
            continue
        if char == '{':  # Start of Comment
            start_line = line
            i += 1
            comment_level = 1 # Support nested comments
            while i < length:
                if code[i] == '{':
                    comment_level += 1
                elif code[i] == '}':
                    comment_level -= 1
                    if comment_level == 0:
                        break
                elif code[i] == '\n':
                    line += 1
                i += 1
            if comment_level != 0:
                tokens.append(('MISMATCH', f"Unclosed comment starting line {start_line}", start_line))
                break # unclosed comment
            i += 1 
            continue

        # --- Operators and Punctuation ---
        # Check for two-character operators first
        if i + 1 < length:
            two_char = code[i:i + 2]
            if two_char in OPERATORS: 
                token_type = 'ASSIGNMENT' if two_char == ASSIGNMENT else 'OPERATOR'
                tokens.append((token_type, two_char, line))
                i += 2
                continue
        # Check for single-character operators and punctuation
        if char in OPERATORS:
            tokens.append(('OPERATOR', char, line))
            i += 1
            continue
        if char in PUNCTUATION: 
            if char not in '{}':
                 tokens.append(('PUNCTUATION', char, line))
            i += 1
            continue
        if char == PARAMETER_PASS: 
            tokens.append(('PARAMETER_PASS', char, line))
            i += 1
            continue

        # --- Identifiers and Keywords ---
        if char.isalpha() or char == '_':
            identifier = ''
            while i < length and (code[i].isalnum() or code[i] == '_'):
                identifier += code[i]
                i += 1
            if identifier in KEYWORDS:
                tokens.append(('KEYWORD', identifier, line))
            elif len(identifier) <= 30: 
                tokens.append(('VARIABLE', identifier, line))
            else: 
                tokens.append(('MISMATCH', f"Identifier '{identifier}' exceeds 30 characters", line))
            continue

        # --- Numbers (Integer and Float) ---
        if char.isdigit() or (char == '.' and i + 1 < length and code[i + 1].isdigit()):
            number = ''
            dot_count = 0
            while i < length and (code[i].isdigit() or code[i] == '.'):
                if code[i] == '.':
                    dot_count += 1
                number += code[i]
                i += 1
            
            if dot_count > 1 or number == '.':
                tokens.append(('MISMATCH', f"Invalid number format '{number}'", line))
            else:
                num_type = "FLOAT" if dot_count == 1 else "INTEGER"
                try:
                    num_value = float(number) if num_type == "FLOAT" else int(number)
                    if num_type == "INTEGER" and (num_value > 32767 or num_value < -32768): 
                        tokens.append(('MISMATCH', f"Integer '{number}' out of allowed range (-32768 to 32767)", line))
                    tokens.append((num_type, num_value, line))
                except ValueError:
                     tokens.append(('MISMATCH', f"Could not convert number '{number}'", line))
                continue

        # --- Mismatched/Unexpected Character ---
        tokens.append(('MISMATCH', f"Unexpected character '{char}'", line))
        i += 1

    return tokens


# --- Symbol Table ---
class SymbolTableError(Exception):
    """Custom exception for symbol table errors."""
    pass

class SymbolTable:
    """
    Manages symbols and scopes for the GreekPlusPlus compiler.
    Handles nested scopes and calculates memory offsets.
    """
    ST_OFFSET_BASE = 12 # Base offset in stack frame for first local/temp/param data

    def __init__(self, sym_file_handle=None):
        """
        Initializes the Symbol Table.

        Args:
            sym_file_handle: An optional file handle to write symbol table dumps.
        """
        self.scopes = [{}]  # List of scope dictionaries (scope 0 is global)
        self.current_level = 0  # Index of the currently active scope during parsing/ICG modification
        # Tracks the *next available* offset within each scope's frame data area
        self.frame_offsets = {0: self.ST_OFFSET_BASE}
        self.sym_file = sym_file_handle

    def get_current_nesting_level(self):
        """Returns the current nesting level (scope index)."""
        return self.current_level

    def open_scope(self):
        """Opens a new nested scope."""
        self.current_level += 1

        # Ensure the scopes list is large enough
        while self.current_level >= len(self.scopes):
            self.scopes.append({})
        # Clear any old data if reusing a scope level index
        self.scopes[self.current_level] = {}
        # Initialize the next available offset for this new scope's frame
        self.frame_offsets[self.current_level] = self.ST_OFFSET_BASE

    def close_scope(self):
        """
        Closes the current scope and optionally writes its contents to the sym_file.

        Returns:
            int: The calculated size of the frame data area (params+locals+temps) for the closed scope.
                 Returns 0 if closing the global scope (which shouldn't happen via this method).

        Raises:
            SymbolTableError: If attempting to close the global scope (level 0).
        """
        if self.current_level <= 0:
            raise SymbolTableError("Cannot explicitly close global scope (level 0).")

        # Log scope details to file before closing
        if self.sym_file:
            scope_to_close = self.scopes[self.current_level]
            header = f"--- Closing Scope Level {self.current_level} ---\n"
            self.sym_file.write(header)
            if scope_to_close:
                # Sort items by offset for clarity before printing
                sorted_items = sorted(scope_to_close.items(),
                                      key=lambda item: item[1].get('offset', float('inf')))
                formatted_scope = pprint.pformat(dict(sorted_items), indent=2, width=100)
                self.sym_file.write(formatted_scope + "\n\n")
            else:
                self.sym_file.write("(Scope was empty)\n\n")

        # Calculate frame data size for the closing scope
        current_scope_final_offset = self.frame_offsets.get(self.current_level, self.ST_OFFSET_BASE)
        # Size is the difference between the final offset and the base offset for data
        calculated_size = current_scope_final_offset - self.ST_OFFSET_BASE
        # Ensure size is non-negative
        calculated_size = max(0, calculated_size)

        self.current_level -= 1
        return calculated_size

    def insert(self, name, entry_data):
        """
        Inserts a new symbol into the current scope.

        Args:
            name (str): The name of the identifier.
            entry_data (dict): A dictionary containing properties of the symbol
                               (e.g., 'entry_type', 'data_type').

        Returns:
            dict: The dictionary representing the inserted symbol entry, augmented
                  with scope level and potentially offset/size.

        Raises:
            SymbolTableError: If the identifier is already declared in the current scope,
                              or if the current scope level is invalid.
        """
        if not (0 <= self.current_level < len(self.scopes)):
            raise SymbolTableError(f"ST Internal Error: Invalid level {self.current_level} for insertion.")

        current_scope_dict = self.scopes[self.current_level]
        current_scope_level = self.current_level

        if name in current_scope_dict:
            raise SymbolTableError(f"Identifier '{name}' already declared in scope {current_scope_level}.")

        # --- Augment entry data ---
        entry_data['scope_level'] = current_scope_level
        entry_data.setdefault('name', name)

        entry_type = entry_data.get('entry_type')

        # Assign offset and update frame size for data-holding entries
        if entry_type in ['variable', 'parameter', 'temp_variable']:
            current_next_offset = self.frame_offsets.get(current_scope_level, self.ST_OFFSET_BASE)
            entry_data.setdefault('offset', current_next_offset)
            entry_data.setdefault('size', 4) # Assume standard size (e.g., 4 bytes for int/pointer)
            self.frame_offsets[current_scope_level] = current_next_offset + entry_data['size']
            entry_data.setdefault('data_type', 'integer') # Default data type if not specified

        # For functions/procedures, initialize structure to hold parameter info later
        elif entry_type in ['function', 'procedure']:
            entry_data.setdefault('parameters', {})
            entry_data.setdefault('start_quad', None) 

        current_scope_dict[name] = entry_data
        return entry_data

    def lookup(self, name, start_level):
        """
        Searches for an identifier starting from a given scope level and going outwards.

        Args:
            name (str): The name of the identifier to look up.
            start_level (int): The scope level (index) to start searching from.

        Returns:
            dict: The dictionary entry for the found symbol.

        Raises:
            SymbolTableError: If the identifier is not found in any accessible scope,
                              or if the start_level is invalid.
        """
        if not (0 <= start_level < len(self.scopes)):
            max_level = len(self.scopes) - 1
            start_level = min(start_level, max_level)
            if start_level < 0:
                 raise SymbolTableError(f"Identifier '{name}' not declared (no valid scopes to search).")

        for level_index in range(start_level, -1, -1):
            if level_index < len(self.scopes):
                scope = self.scopes[level_index]
                if name in scope:
                    return scope[name]

        raise SymbolTableError(f"Identifier '{name}' not declared in accessible scopes (searched from level {start_level} down to 0).")

    def update_entry(self, name, scope_level, **kwargs):
        """
        Updates properties of an existing symbol entry in a specific scope.

        Args:
            name (str): The name of the identifier.
            scope_level (int): The scope level where the identifier is defined.
            **kwargs: Key-value pairs representing the properties to update.

        Raises:
            SymbolTableError: If the scope level is invalid or the identifier is not found
                              at that level.
        """
        if not (0 <= scope_level < len(self.scopes)):
            raise SymbolTableError(f"Cannot update '{name}': Invalid scope level {scope_level}")

        scope = self.scopes[scope_level]
        if name not in scope:
            raise SymbolTableError(f"Cannot update '{name}': Identifier not found in scope {scope_level}")

        scope[name].update(kwargs)

    def get_current_scope_dict(self):
        """Returns the dictionary for the currently active scope."""
        if 0 <= self.current_level < len(self.scopes):
            return self.scopes[self.current_level]
        else:
            return {}

# --- Parser ---
class GreekPlusPlusParser:
    """
    Parses the token stream generated by the lexer, builds an Abstract Syntax Tree,
    and interacts with the Symbol Table for semantic analysis (declarations, lookups).
    """
    def __init__(self, tokens, sym_file_handle=None):
        """
        Initializes the parser.

        Args:
            tokens (list): The list of tokens from the lexer.
            sym_file_handle: An optional file handle for the SymbolTable.
        """
        self.tokens = list(tokens)
        self.pos = 0
        self.current_token = self.tokens[self.pos] if self.tokens else None
        self.symbol_table = SymbolTable(sym_file_handle=sym_file_handle)
        self.line_number = 1
        self.update_line_number()
        self.current_defining_block_entry = None

    def update_line_number(self):
        """Updates the current line number based on the current token."""
        if self.current_token and len(self.current_token) >= 3:
            self.line_number = self.current_token[2]

    def advance(self):
        """Moves to the next token in the stream."""
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current_token = self.tokens[self.pos]
            self.update_line_number()
        else:
            self.current_token = None

    def peek(self):
        """Returns the next token without consuming the current one."""
        peek_pos = self.pos + 1
        return self.tokens[peek_pos] if peek_pos < len(self.tokens) else None

    def raise_error(self, message):
        """Raises a SyntaxError with context information."""
        tok_str = f"'{self.current_token[1]}'" if self.current_token else "EOF"
        tok_type = self.current_token[0] if self.current_token else "N/A"
        raise SyntaxError(f"{message} near {tok_type} token {tok_str} at line {self.line_number}")

    def expect(self, expected_type, expected_value=None):
        """
        Consumes the current token if it matches the expected type and optional value.

        Args:
            expected_type (str): The expected token type (e.g., 'KEYWORD', 'VARIABLE').
            expected_value (str, optional): The expected token value (e.g., 'πρόγραμμα').

        Returns:
            tuple: The consumed token.

        Raises:
            SyntaxError: If the current token does not match the expectation.
        """
        token = self.current_token
        if not token:
            self.raise_error(f"Expected {expected_type}{' ('+expected_value+')' if expected_value else ''}, but found End of File")
        if token[0] != expected_type:
            self.raise_error(f"Expected token type {expected_type}, but found {token[0]} ('{token[1]}')")
        if expected_value is not None and token[1] != expected_value:
            self.raise_error(f"Expected '{expected_value}', but found '{token[1]}'")

        self.advance()
        return token

    def parse(self):
        """Starts the parsing process for the entire program."""
        try:
            ast = self.program()
            if self.current_token:
                # If there are tokens left after parsing the program, it's an error
                self.raise_error("Unexpected token found after the end of the program")
            return ast
        except (SyntaxError, SymbolTableError) as e:
            print(f"\nCOMPILATION FAILED: {e}", file=sys.stderr)
            raise 
        except Exception as e:
            print(f"\nINTERNAL PARSER ERROR near line {self.line_number}: {e}", file=sys.stderr)
            raise

    # --- Grammar Rule Implementations ---

    def program(self):
        """ Parses: πρόγραμμα <id> declarations subprograms αρχή_προγράμματος statements τέλος_προγράμματος """
        self.expect('KEYWORD', 'πρόγραμμα')
        name_tok = self.expect('VARIABLE')
        program_name = name_tok[1]

        program_entry = self.symbol_table.insert(program_name, {'entry_type': 'program'})
        self.current_defining_block_entry = program_entry

        decls_ast = self.declarations()
        for var_list in decls_ast:
            for var_name in var_list:
                try:
                    self.symbol_table.insert(var_name, {'entry_type': 'variable'})
                except SymbolTableError as e:
                    self.raise_error(f"Declaration error in program '{program_name}': {e}")

        subs_ast = self.subprograms()

        self.expect('KEYWORD', 'αρχή_προγράμματος')
        stmts_ast = self.statement_sequence(end_keywords={'τέλος_προγράμματος'})
        self.expect('KEYWORD', 'τέλος_προγράμματος')

        # Store frame offset info for main program's scope (level 0) in its ST entry.
        self.symbol_table.update_entry(program_name, 0, frame_offsets_info=self.symbol_table.frame_offsets)

        return ('PROGRAM', program_name, decls_ast, subs_ast, stmts_ast)

    def declarations(self):
        """ Parses: { δήλωση varlist } """
        declarations_list = []
        while self.current_token and self.current_token[:2] == ('KEYWORD', 'δήλωση'):
            self.advance()
            variables = self.varlist()
            if not variables:
                self.raise_error("Expected one or more variable names after 'δήλωση'")
            declarations_list.append(variables)
            # Optional semicolon after declaration? Grammar doesn't specify, assuming not required.
        return declarations_list

    def varlist(self):
        """ Parses: <id> { , <id> } """
        variables = []
        if not self.current_token or self.current_token[0] != 'VARIABLE':
            # It's valid to have no variables if the context allows (e.g., empty param list)
            return variables

        while True:
            var_tok = self.expect('VARIABLE')
            variables.append(var_tok[1])
            if self.current_token and self.current_token[:2] == ('PUNCTUATION', ','):
                self.advance()
                if not self.current_token or self.current_token[0] != 'VARIABLE':
                    self.raise_error("Expected a variable name after comma in variable list")
            else:
                break
        return variables

    def subprograms(self):
        """ Parses: { ( function | procedure ) } """
        subprogram_list = []
        while self.current_token and self.current_token[0] == 'KEYWORD' and \
              self.current_token[1] in ['συνάρτηση', 'διαδικασία']:
            if self.current_token[1] == 'συνάρτηση':
                subprogram_list.append(self.function())
            else:
                subprogram_list.append(self.procedure())
        return subprogram_list

    def _parse_subprogram_header(self, subprogram_type):
        """ Helper to parse the common header part of functions and procedures. """
        self.expect('KEYWORD', subprogram_type)
        name_tok = self.expect('VARIABLE')
        name = name_tok[1]

        original_defining_block_entry = self.current_defining_block_entry
        definition_level = self.symbol_table.get_current_nesting_level()

        entry_type = 'function' if subprogram_type == 'συνάρτηση' else 'procedure'
        try:
            new_block_st_entry = self.symbol_table.insert(name, {'entry_type': entry_type, 'name': name})
        except SymbolTableError as e:
            self.raise_error(f"Error declaring {subprogram_type} '{name}': {e}")

        self.current_defining_block_entry = new_block_st_entry

        self.symbol_table.open_scope()
        body_and_param_scope_level = self.symbol_table.get_current_nesting_level()

        if entry_type == 'function':
            try:
                self.symbol_table.insert(name, {
                    'entry_type': 'variable',
                    'data_type': 'integer', 
                    'is_return_var': True
                })
            except SymbolTableError as e:
                 self.raise_error(f"Internal error creating return variable for function '{name}': {e}")

        self.expect('PUNCTUATION', '(')
        params_ast, param_names = self.parameters()
        self.expect('PUNCTUATION', ')')

        self.expect('KEYWORD', 'διαπροσωπεία')
        modes_ast, param_details_for_st_entry = self.parameter_modes(param_names)

        self.symbol_table.update_entry(name, definition_level, parameters=param_details_for_st_entry)

        return name, params_ast, modes_ast, definition_level, original_defining_block_entry, body_and_param_scope_level

    def function(self):
        """ Parses: συνάρτηση <id> ( parameters ) διαπροσωπεία parameter_modes αρχή_συνάρτησης statements τέλος_συνάρτησης """
        name, params_ast, modes_ast, definition_level, original_defining_block_entry, body_scope_level = \
            self._parse_subprogram_header('συνάρτηση')

        start_keyword = 'αρχή_συνάρτησης'
        end_keyword = 'τέλος_συνάρτησης'
        self.expect('KEYWORD', start_keyword)

        stmts_ast = self.statement_sequence(end_keywords={end_keyword})

        self.expect('KEYWORD', end_keyword)

        self.symbol_table.close_scope()
        self.current_defining_block_entry = original_defining_block_entry
        return ('FUNCTION', name, params_ast, modes_ast, stmts_ast, definition_level)

    def procedure(self):
        """ Parses: διαδικασία <id> ( parameters ) διαπροσωπεία parameter_modes αρχή_διαδικασίας statements τέλος_διαδικασίας """
        name, params_ast, modes_ast, definition_level, original_defining_block_entry, body_scope_level = \
            self._parse_subprogram_header('διαδικασία')

        start_keyword = 'αρχή_διαδικασίας'
        end_keyword = 'τέλος_διαδικασίας'
        self.expect('KEYWORD', start_keyword)

        stmts_ast = self.statement_sequence(end_keywords={end_keyword})

        self.expect('KEYWORD', end_keyword)

        self.symbol_table.close_scope()
        self.current_defining_block_entry = original_defining_block_entry
        return ('PROCEDURE', name, params_ast, modes_ast, stmts_ast, definition_level)

    def parameters(self):
        """ Parses: [ <id> { , <id> } ] """
        param_name_list = []
        param_ast_nodes = []
        if self.current_token and self.current_token[0] == 'VARIABLE':
            while True:
                name_tok = self.expect('VARIABLE')
                name = name_tok[1]
                param_name_list.append(name)
                param_ast_nodes.append(('PARAM_NAME', name))
                if self.current_token and self.current_token[:2] == ('PUNCTUATION', ','):
                    self.advance()
                    if not self.current_token or self.current_token[0] != 'VARIABLE':
                        self.raise_error("Expected parameter name after comma in parameter list")
                else:
                    break
        return param_ast_nodes, param_name_list

    def parameter_modes(self, declared_param_names):
        """
        Parses: [ είσοδος varlist ] [ έξοδος varlist ]
        Also inserts parameters into the current symbol table scope (function/proc body scope).
        """
        modes_ast = {'είσοδος': [], 'έξοδος': []}
        param_details_for_parent = {}
        processed_params = set()
        param_order_map = {name: i for i, name in enumerate(declared_param_names)}

        if self.current_token and self.current_token[:2] == ('KEYWORD', 'είσοδος'):
            self.advance()
            cv_names = self.varlist()
            if not cv_names:
                self.raise_error("Variable list expected after 'είσοδος'")
            modes_ast['είσοδος'] = cv_names
            for name in cv_names:
                if name not in declared_param_names:
                    self.raise_error(f"Parameter '{name}' in 'είσοδος' was not declared in parentheses.")
                if name in processed_params:
                    self.raise_error(f"Parameter '{name}' declared with multiple modes ('είσοδος' and 'έξοδος').")

                try:
                    param_entry = self.symbol_table.insert(name, {
                        'entry_type': 'parameter',
                        'mode': 'CV',
                        'order': param_order_map[name],
                        'data_type': 'integer' # Assuming integer parameters for now
                    })
                    param_details_for_parent[name] = {
                        'mode': 'CV',
                        'order': param_order_map[name],
                        'offset': param_entry['offset'],
                        'type': param_entry['data_type'],
                        'name': name
                    }
                    processed_params.add(name)
                except SymbolTableError as e:
                     self.raise_error(f"Error declaring 'είσοδος' parameter '{name}': {e}")

        if self.current_token and self.current_token[:2] == ('KEYWORD', 'έξοδος'):
            self.advance()
            ref_names = self.varlist()
            if not ref_names:
                self.raise_error("Variable list expected after 'έξοδος'")
            modes_ast['έξοδος'] = ref_names
            for name in ref_names:
                if name not in declared_param_names:
                    self.raise_error(f"Parameter '{name}' in 'έξοδος' was not declared in parentheses.")
                if name in processed_params:
                    self.raise_error(f"Parameter '{name}' declared with multiple modes ('είσοδος' and 'έξοδος').")

                try:
                    param_entry = self.symbol_table.insert(name, {
                        'entry_type': 'parameter',
                        'mode': 'REF',
                        'order': param_order_map[name],
                        'data_type': 'integer' 
                    })
                    param_details_for_parent[name] = {
                        'mode': 'REF',
                        'order': param_order_map[name],
                        'offset': param_entry['offset'],
                        'type': param_entry['data_type'],
                        'name': name
                    }
                    processed_params.add(name)
                except SymbolTableError as e:
                     self.raise_error(f"Error declaring 'έξοδος' parameter '{name}': {e}")

        if len(processed_params) != len(declared_param_names):
            missing_params = set(declared_param_names) - processed_params
            self.raise_error(f"Parameter(s) {missing_params} were declared but missing mode ('είσοδος' or 'έξοδος').")

        return ('PARAM_MODES', modes_ast), param_details_for_parent

    def _parse_argument_list(self, callable_name, callable_entry):
        """ Helper to parse argument lists for function calls and execute statements. """
        args_ast_nodes = []
        passed_arg_count = 0

        expected_params_dict = callable_entry.get('parameters', {})
        expected_param_list = sorted(expected_params_dict.values(), key=lambda item: item['order'])

        self.expect('PUNCTUATION', '(')

        if not (self.current_token and self.current_token[1] == ')'):
            while True:
                if passed_arg_count >= len(expected_param_list):
                    self.raise_error(f"Too many arguments provided for call to '{callable_name}'. Expected {len(expected_param_list)}.")

                formal_param_info = expected_param_list[passed_arg_count]
                formal_param_name = formal_param_info['name']
                expected_mode = formal_param_info.get('mode', 'CV')

                is_ref_attempt = False
                if self.current_token and self.current_token[0] == 'PARAMETER_PASS':
                    self.advance()
                    is_ref_attempt = True

                arg_expr_tokens = self.expression()
                if not arg_expr_tokens:
                    # double-check Missing expression
                    self.raise_error(f"Missing expression for argument {passed_arg_count + 1} ('{formal_param_name}') in call to '{callable_name}'.")

                if expected_mode == 'REF' and not is_ref_attempt:
                    self.raise_error(f"Argument {passed_arg_count + 1} ('{formal_param_name}') for '{callable_name}' expects call-by-reference (REF), but '%' was not used in the call.")
                if expected_mode == 'REF' and is_ref_attempt and \
                   not (len(arg_expr_tokens) == 1 and arg_expr_tokens[0][0] == 'VARIABLE'):
                    self.raise_error(f"Argument {passed_arg_count + 1} ('{formal_param_name}') for '{callable_name}' passed by reference ('%') must be a single variable name, not an expression.")
                if expected_mode == 'CV' and is_ref_attempt:
                     self.raise_error(f"Argument {passed_arg_count + 1} ('{formal_param_name}') for '{callable_name}' expects call-by-value (CV), but '%' was used.")

                args_ast_nodes.append({
                    'expression': arg_expr_tokens,
                    'mode': expected_mode,
                    'is_ref_token_present': is_ref_attempt
                })
                passed_arg_count += 1

                if self.current_token and self.current_token[1] == ',':
                    self.advance()
                    if self.current_token and self.current_token[1] == ')':
                        self.raise_error(f"Unexpected ')' after comma in argument list for '{callable_name}'.")
                elif self.current_token and self.current_token[1] == ')':
                    break
                else:
                    tok_info = f"'{self.current_token[1]}'" if self.current_token else "EOF"
                    self.raise_error(f"Expected ',' or ')' after argument {passed_arg_count} for '{callable_name}', but found {tok_info}.")

        self.expect('PUNCTUATION', ')')

        if passed_arg_count != len(expected_param_list):
            self.raise_error(f"Call to '{callable_name}' expects {len(expected_param_list)} argument(s), but {passed_arg_count} were provided.")

        return args_ast_nodes

    def function_call(self):
        """ Parses a function call used within an expression: <id> ( arguments ) """
        name_tok = self.expect('VARIABLE')
        name = name_tok[1]

        try:
            lookup_level = self.symbol_table.get_current_nesting_level()
            entry = self.symbol_table.lookup(name, lookup_level)
        except SymbolTableError as e:
            self.raise_error(f"Function Call Error: {e}")

        if entry['entry_type'] != 'function':
            self.raise_error(f"Identifier '{name}' is a {entry['entry_type']}, not a function. Cannot call it here.")

        args_ast_nodes = self._parse_argument_list(name, entry)
        return ('FUNCTION_CALL', name, args_ast_nodes)

    def statement_sequence(self, end_keywords):
        """
        Parses a sequence of statements until an end keyword or EOF is reached.
        Handles optional semicolons between statements.
        """
        statements = []
        end_keywords_set = set(end_keywords) if not isinstance(end_keywords, set) else end_keywords

        while True:
            while self.current_token and self.current_token[:2] == ('PUNCTUATION', ';'):
                self.advance()

            if not self.current_token:
                break
            if self.current_token[0] == 'KEYWORD' and self.current_token[1] in end_keywords_set:
                break

            stmt_ast = self.statement()
            if stmt_ast:
                statements.append(stmt_ast)
        return statements

    def statement(self):
        """ Parses a single statement based on the starting token. """
        if not self.current_token:
            self.raise_error("Expected a statement, but reached End of File.")

        tok_type, tok_val = self.current_token[0], self.current_token[1]

        if tok_type == 'VARIABLE':
            next_token = self.peek()
            if next_token and next_token[0] == 'ASSIGNMENT':
                return self.assignment_statement()
            else:
                self.raise_error(f"Expected ':=' after variable '{tok_val}' to start an assignment statement, or use 'εκτέλεσε' for procedure calls.")

        elif tok_type == 'KEYWORD':
            if tok_val == 'εάν': 
                return self.if_statement()
            if tok_val == 'όσο': 
                return self.while_loop()
            if tok_val == 'για': 
                return self.for_loop()
            if tok_val == 'επανάλαβε': 
                return self.repeat_until_loop()
            if tok_val == 'διάβασε': 
                return self.read_statement()
            if tok_val == 'γράψε': 
                return self.print_statement()
            if tok_val == 'εκτέλεσε': 
                return self.execute_statement()
            self.raise_error(f"Unexpected keyword '{tok_val}' starting a statement.")

        elif tok_type == 'PUNCTUATION' and tok_val == ';':
            self.advance()
            return None

        else:
            self.raise_error(f"Unexpected token '{tok_val}' ({tok_type}) found where a statement was expected.")
        return None

    def assignment_statement(self):
        """ Parses: <variable> := expression """
        var_tok = self.expect('VARIABLE')
        var_name = var_tok[1]
    
        try:
            lookup_level = self.symbol_table.get_current_nesting_level()
            entry = self.symbol_table.lookup(var_name, lookup_level)

            is_assignable = False
            if entry['entry_type'] == 'variable':
                is_return_var_for_current_func = (
                    entry.get('is_return_var', False) and
                    self.current_defining_block_entry and
                    self.current_defining_block_entry['entry_type'] == 'function' and
                    self.current_defining_block_entry['name'] == var_name and
                    entry['scope_level'] == self.symbol_table.get_current_nesting_level()
                )
                if not entry.get('is_return_var', False) or is_return_var_for_current_func:
                    is_assignable = True
            elif entry['entry_type'] == 'parameter' and entry.get('mode') == 'REF':
                is_assignable = True

            if not is_assignable:
                 entry_desc = f"{entry['entry_type']}"
                 if entry.get('entry_type') == 'parameter': 
                     entry_desc += f" (mode: {entry.get('mode', 'N/A')})"
                 if entry.get('entry_type') == 'function': 
                     entry_desc += " (function name)"
                 self.raise_error(f"Cannot assign to '{var_name}'. It is not a modifiable variable or REF parameter in this context (it's a {entry_desc}).")

        except SymbolTableError as e:
            self.raise_error(f"Assignment Error: {e}")

        self.expect('ASSIGNMENT', ':=')
        expr_tokens = self.expression()
        if not expr_tokens:
            self.raise_error(f"Missing expression on the right side of assignment to '{var_name}'.")

        return ('ASSIGNMENT', var_name, expr_tokens)

    def expression(self):
        """
        Collects a sequence of tokens that form an expression.
        Does not build an expression AST here, relies on ICG to parse the token list.
        Handles detection of standalone function calls within the token stream.
        Performs basic semantic checks (e.g., variable lookup).
        """
        tokens_for_expr = []
        paren_nesting_level = 0
        bracket_nesting_level = 0

        if self.current_token and self.current_token[0] == 'VARIABLE' and \
           self.peek() and self.peek()[1] == '(':
            start_pos_backup = self.pos
            current_token_backup = self.current_token
            try:
                call_node = self.function_call()
                if not self.current_token or \
                   not (self.current_token[0] in ['OPERATOR', 'KEYWORD'] and
                        self.current_token[1] in OP_PRECEDENCE and OP_PRECEDENCE.get(self.current_token[1],-1) > 0):
                    return [call_node]
                else:
                    self.pos = start_pos_backup
                    self.current_token = current_token_backup
                    self.update_line_number()
            except (SyntaxError, SymbolTableError):
                self.pos = start_pos_backup
                self.current_token = current_token_backup
                self.update_line_number()

        expression_terminators = {
            ('KEYWORD', 'τότε'), ('KEYWORD', 'έως'), ('KEYWORD', 'με_βήμα'),
            ('KEYWORD', 'μέχρι'), ('KEYWORD', 'επανάλαβε'),
            ('KEYWORD', 'εάν_τέλος'), ('KEYWORD', 'όσο_τέλος'), ('KEYWORD', 'για_τέλος'),
            ('KEYWORD', 'τέλος_προγράμματος'), ('KEYWORD', 'τέλος_συνάρτησης'),
            ('KEYWORD', 'τέλος_διαδικασίας'), ('KEYWORD', 'αλλιώς'),
            ('PUNCTUATION', ';'), ('PUNCTUATION', ','),
            ('ASSIGNMENT', ':=')
        }

        while self.current_token:
            tok_type, tok_val, tok_line = self.current_token[0], self.current_token[1], self.current_token[2]
            current_token_tuple = (tok_type, tok_val)

            if paren_nesting_level == 0 and bracket_nesting_level == 0:
                if current_token_tuple in expression_terminators:
                    break
                if tok_val in [')', ']']:
                    break

            if tok_val == '(': 
                paren_nesting_level += 1
            elif tok_val == ')':
                if paren_nesting_level == 0: 
                    break
                paren_nesting_level -= 1
            elif tok_val == '[': 
                bracket_nesting_level += 1
            elif tok_val == ']':
                if bracket_nesting_level == 0: 
                    break
                bracket_nesting_level -= 1

            if tok_type == 'VARIABLE':
                try:
                    lookup_level = self.symbol_table.get_current_nesting_level()
                    self.symbol_table.lookup(tok_val, lookup_level)
                except SymbolTableError as e:
                    self.raise_error(f"Expression Error: {e}")
            elif tok_type == 'MISMATCH':
                 self.raise_error(f"Invalid token '{tok_val}' found within expression.")

            tokens_for_expr.append(self.current_token)
            self.advance()

        if paren_nesting_level != 0:
            self.raise_error("Mismatched parentheses '(' / ')' detected in expression.")
        if bracket_nesting_level != 0:
            self.raise_error("Mismatched brackets '[' / ']' detected in expression.")

        if not tokens_for_expr:
            next_tok_info = f"'{self.current_token[1]}'" if self.current_token else "EOF"
            self.raise_error(f"Expected an expression, but found none (or expression terminated immediately by {next_tok_info}).")

        return tokens_for_expr


    def if_statement(self):
        """ Parses: εάν expression τότε statement_sequence [ αλλιώς statement_sequence ] εάν_τέλος """
        self.expect('KEYWORD', 'εάν')
        condition_expr_tokens = self.expression()
        if not condition_expr_tokens:
            self.raise_error("Missing condition after 'εάν'.")

        self.expect('KEYWORD', 'τότε')
        then_block_stmts = self.statement_sequence(end_keywords={'αλλιώς', 'εάν_τέλος'})

        else_block_stmts = []
        if self.current_token and self.current_token[:2] == ('KEYWORD', 'αλλιώς'):
            self.advance()
            else_block_stmts = self.statement_sequence(end_keywords={'εάν_τέλος'})

        self.expect('KEYWORD', 'εάν_τέλος')
        return ('IF', condition_expr_tokens, then_block_stmts, else_block_stmts)

    def while_loop(self):
        """ Parses: όσο expression επανάλαβε statement_sequence όσο_τέλος """
        self.expect('KEYWORD', 'όσο')
        condition_expr_tokens = self.expression()
        if not condition_expr_tokens:
            self.raise_error("Missing condition after 'όσο'.")

        self.expect('KEYWORD', 'επανάλαβε')
        loop_body_stmts = self.statement_sequence(end_keywords={'όσο_τέλος'})
        self.expect('KEYWORD', 'όσο_τέλος')
        return ('WHILE', condition_expr_tokens, loop_body_stmts)

    def repeat_until_loop(self):
        """ Parses: επανάλαβε statement_sequence μέχρι expression """
        self.expect('KEYWORD', 'επανάλαβε')
        loop_body_stmts = self.statement_sequence(end_keywords={'μέχρι'})

        self.expect('KEYWORD', 'μέχρι')
        condition_expr_tokens = self.expression()
        if not condition_expr_tokens:
            self.raise_error("Missing condition after 'μέχρι'.")
        # Note: Condition is evaluated *after* the loop body executes at least once.
        return ('REPEAT_UNTIL', loop_body_stmts, condition_expr_tokens)

    def for_loop(self):
        """ Parses: για <id> := expression έως expression [ με_βήμα expression ] επανάλαβε statement_sequence για_τέλος """
        self.expect('KEYWORD', 'για')
        loop_var_tok = self.expect('VARIABLE')
        loop_var_name = loop_var_tok[1]

        try:
            lookup_level = self.symbol_table.get_current_nesting_level()
            entry = self.symbol_table.lookup(loop_var_name, lookup_level)
            if entry['entry_type'] not in ['variable', 'parameter']:
                 self.raise_error(f"FOR loop counter '{loop_var_name}' must be a simple variable or parameter, not a {entry['entry_type']}.")
        except SymbolTableError:
            self.raise_error(f"FOR loop counter variable '{loop_var_name}' not declared.")

        self.expect('ASSIGNMENT', ':=')
        start_expr_tokens = self.expression()
        if not start_expr_tokens: 
            self.raise_error("Missing start value expression in FOR loop.")

        self.expect('KEYWORD', 'έως')
        end_expr_tokens = self.expression()
        if not end_expr_tokens: 
            self.raise_error("Missing end value expression in FOR loop.")

        step_expr_tokens = [('INTEGER', 1, self.line_number)]
        if self.current_token and self.current_token[:2] == ('KEYWORD', 'με_βήμα'):
            self.advance()
            step_expr_tokens = self.expression()
            if not step_expr_tokens: 
                self.raise_error("Missing step value expression after 'με_βήμα'.")

        self.expect('KEYWORD', 'επανάλαβε')
        loop_body_stmts = self.statement_sequence(end_keywords={'για_τέλος'})
        self.expect('KEYWORD', 'για_τέλος')
        return ('FOR', loop_var_name, start_expr_tokens, end_expr_tokens, step_expr_tokens, loop_body_stmts)

    def read_statement(self):
        """ Parses: διάβασε varlist """
        self.expect('KEYWORD', 'διάβασε')
        variables_to_read = self.varlist()
        if not variables_to_read:
            self.raise_error("Expected one or more variable names after 'διάβασε'.")

        for var_name in variables_to_read:
            try:
                lookup_level = self.symbol_table.get_current_nesting_level()
                entry = self.symbol_table.lookup(var_name, lookup_level)
                if entry['entry_type'] not in ['variable'] and \
                   not (entry['entry_type'] == 'parameter' and entry.get('mode') == 'REF'):
                    self.raise_error(f"Cannot read into '{var_name}'. It must be a modifiable variable or a REF parameter, not a {entry['entry_type']}.")
            except SymbolTableError :
                self.raise_error(f"Variable '{var_name}' in 'διάβασε' statement not declared.")
        return ('READ', variables_to_read)

    def print_statement(self):
        """ Parses: γράψε expression { , expression } """
        self.expect('KEYWORD', 'γράψε')
        expressions_to_print = []

        if not self.current_token or \
           (self.current_token[0] == 'PUNCTUATION' and self.current_token[1] == ';') or \
           (self.current_token[0] == 'KEYWORD'):
            self.raise_error("Expression expected after 'γράψε'.")

        while True:
            expr_tokens = self.expression()
            if not expr_tokens:
                # Should be caught by expression(), but double-check
                self.raise_error("Missing expression in 'γράψε' statement.")
            expressions_to_print.append(expr_tokens)

            if self.current_token and self.current_token[:2] == ('PUNCTUATION', ','):
                self.advance()
                if not self.current_token or \
                   (self.current_token[0] == 'PUNCTUATION' and self.current_token[1] == ';') or \
                   (self.current_token[0] == 'KEYWORD'):
                    self.raise_error("Expression expected after comma in 'γράψε' statement.")
            else:
                break
        return ('PRINT', expressions_to_print)


    def execute_statement(self):
        """ Parses: εκτέλεσε <id> ( arguments ) """
        self.expect('KEYWORD', 'εκτέλεσε')
        name_tok = self.expect('VARIABLE')
        name = name_tok[1]

        try:
            lookup_level = self.symbol_table.get_current_nesting_level()
            entry = self.symbol_table.lookup(name, lookup_level)
        except SymbolTableError as e:
            self.raise_error(f"Execute Error: {e}")

        if entry['entry_type'] != 'procedure':
            self.raise_error(f"Identifier '{name}' is a {entry['entry_type']}, not a procedure. Use 'εκτέλεσε' only for procedures.")

        args_ast_nodes = self._parse_argument_list(name, entry)
        return ('EXECUTE', name, args_ast_nodes)


# --- Intermediate Code Generator ---
class IntermediateCodeGenerator:
    """
    Generates intermediate code (quadruples) from the Abstract Syntax Tree,
    using the Symbol Table for context and temporary variable management.
    """
    def __init__(self, symbol_table):
        """
        Initializes the ICG.

        Args:
            symbol_table (SymbolTable): The symbol table populated by the parser.
        """
        self.symbol_table = symbol_table
        self.quads = []
        self.temp_count = 1
        self.quad_label = 1
        self.expr_token_list = []
        self.expr_pos = 0
        self.current_expr_token = None
        self.current_generating_block_st_entry = None
        self.gen_body_scope_level = -1

    def _set_gen_context(self, block_st_entry):
        """Sets the ICG's context to the given block (program, function, or procedure)."""
        self.current_generating_block_st_entry = block_st_entry
        if not block_st_entry:
            self.gen_body_scope_level = -1
            return

        if block_st_entry['entry_type'] == 'program':
            self.gen_body_scope_level = 0
        else:
            definition_level = block_st_entry.get('scope_level', -1)
            self.gen_body_scope_level = definition_level + 1

    def _restore_gen_context(self, original_block_st_entry, original_gen_body_scope_level):
        """Restores the ICG's context to a previous state."""
        self.current_generating_block_st_entry = original_block_st_entry
        self.gen_body_scope_level = original_gen_body_scope_level

    def nextquad(self):
        """Returns the label for the next quadruple to be generated."""
        return self.quad_label

    def new_temp(self, data_type='integer', target_scope_level=None):
        """
        Creates a new temporary variable, adds it to the symbol table in the
        appropriate scope, and returns its name.
        """
        temp_name = f"t@{self.temp_count}"
        self.temp_count += 1

        effective_target_scope = target_scope_level if target_scope_level is not None else self.gen_body_scope_level

        if effective_target_scope < 0:
             raise ValueError(f"ICG Error: Invalid target scope level ({effective_target_scope}) for inserting temporary '{temp_name}'. Current block: {self.current_generating_block_st_entry.get('name', 'None') if self.current_generating_block_st_entry else 'None'}")

        original_st_active_level = self.symbol_table.current_level
        self.symbol_table.current_level = effective_target_scope

        try:
            self.symbol_table.insert(temp_name, {
                'entry_type': 'temp_variable',
                'data_type': data_type
            })
        except SymbolTableError:
            pass 
        finally:
            self.symbol_table.current_level = original_st_active_level

        return temp_name

    def emit(self, op, arg1, arg2, result):
        """
        Creates a quadruple and adds it to the list of quads.
        Args:
            op (str): The operator/instruction.
            arg1: The first argument.
            arg2: The second argument.
            result: The result/target.
        Returns:
            int: The label of the emitted quadruple.
        """
        op_str = str(op if op is not None else '_')
        arg1_str = str(arg1 if arg1 is not None else '_')
        arg2_str = str(arg2 if arg2 is not None else '_')
        result_str = str(result if result is not None else '_')

        quad = {
            'label': self.quad_label,
            'op': op_str,
            'arg1': arg1_str,
            'arg2': arg2_str,
            'result': result_str
        }
        self.quads.append(quad)
        current_label = self.quad_label
        self.quad_label += 1
        return current_label

    # --- Backpatching List Management ---
    def emptylist(self):
        """Returns an empty list, used for initializing backpatch lists."""
        return []

    def makelist(self, label):
        """
        Creates a new list containing a single quadruple label for backpatching.
        Args:
            label (int): The quadruple label to add to the list.
        Returns:
            list: A list containing the label, or an empty list if label is invalid.
        """
        return [label] if isinstance(label, int) and label > 0 else []

    def mergelist(self, *lists):
        """
        Merges multiple backpatch lists into a single list, removing duplicates.
        Args:
            *lists: A variable number of lists (each a backpatch list).
        Returns:
            list: A sorted list of unique, valid quadruple labels.
        """
        merged = set()
        for lst in lists:
            if isinstance(lst, list):
                merged.update(item for item in lst if isinstance(item, int) and item > 0)
        return sorted(list(merged))

    def backpatch(self, label_list, target_label):
        """
        Fills in the target label ('result' field) for all quads whose labels are in the list.
        Args:
            label_list (list): List of quad labels (integers) needing backpatching.
            target_label (int): The actual target label to jump to.
        """
        if not label_list or not isinstance(target_label, int) or target_label <= 0:
            return

        target_label_str = str(target_label)
        for label_to_patch in label_list:
            if not isinstance(label_to_patch, int) or label_to_patch <= 0:
                continue

            quad_index = label_to_patch - 1
            if 0 <= quad_index < len(self.quads):
                quad = self.quads[quad_index]
                if quad['result'] == '?' or quad['result'] == '_':
                    quad['result'] = target_label_str

       
    # --- Expression Parsing (using Recursive Descent with Precedence) ---

    def _init_expr_parser(self, tokens):
        """
        Initializes the state for parsing a list of expression tokens.
        Args:
            tokens (list): The list of tokens forming the expression.
        """
        self.expr_token_list = tokens
        self.expr_pos = 0
        self.current_expr_token = self.expr_token_list[0] if self.expr_token_list else None

    def _advance_expr_token(self):
        """Moves to the next token in the current expression being parsed."""
        self.expr_pos += 1
        self.current_expr_token = self.expr_token_list[self.expr_pos] if self.expr_pos < len(self.expr_token_list) else None

    def _get_op_precedence(self, token):
        """
        Returns the precedence level of an operator token.
        Args:
            token (tuple): The token to check.
        Returns:
            int: The precedence level, or -1 if not a recognized operator.
        """
        if token and token[0] in ['OPERATOR', 'KEYWORD']:
            return OP_PRECEDENCE.get(token[1], -1)
        return -1

    def _ensure_boolean(self, result, context_op, line_num='?'):
        """
        Ensures an evaluation result represents a boolean value (has truelist/falselist).
        If not (e.g., it's a 'place'), generates code to convert it.
        Args:
            result (dict): The result from a sub-expression evaluation.
            context_op (str): The operator requiring boolean operands (e.g., 'και', 'ή').
            line_num (int/str): Line number for error messages.
        Returns:
            dict: A dictionary with 'truelist' and 'falselist'.
        """
        if isinstance(result, dict) and 'truelist' in result and 'falselist' in result:
            return result
        elif isinstance(result, dict) and 'place' in result:
            place = result['place']
            true_list = self.makelist(self.emit('!=', place, 0, '?'))
            false_list = self.makelist(self.emit('jump', '_', '_', '?'))
            return {'truelist': true_list, 'falselist': false_list}
        else:
            raise ValueError(f"Type Error near line {line_num}: Operator '{context_op}' requires boolean operands, but received incompatible result: {result}")

    def _place_from_eval_result(self, eval_result, context_msg):
        """
        Extracts a 'place' (variable/temp/literal) from an evaluation result.
        If the result is boolean (truelist/falselist), generates code to store
        1 (true) or 0 (false) into a new temporary and returns that temporary's name.
        Args:
            eval_result (dict): Result from expression evaluation.
            context_msg (str): Description of where the place is needed (for errors).
        Returns:
            str: The name of the variable/temporary or the literal value holding the result.
        """
        if isinstance(eval_result, dict) and 'place' in eval_result:
             place = eval_result['place']
             if isinstance(place, str) and 'ERR' in place: # Check for propagated errors
                 raise ValueError(f"Error propagated in place for {context_msg}: {place}")
             return place
        elif isinstance(eval_result, dict) and 'truelist' in eval_result and 'falselist' in eval_result:
             temp_for_bool = self.new_temp()
             true_target_label = self.nextquad()
             self.backpatch(eval_result['truelist'], true_target_label)
             self.emit(':=', 1, '_', temp_for_bool)
             goto_after_bool_assign = self.makelist(self.emit('jump', '_', '_', '?'))
             false_target_label = self.nextquad()
             self.backpatch(eval_result['falselist'], false_target_label)
             self.emit(':=', 0, '_', temp_for_bool)
             after_bool_assign_label = self.nextquad()
             self.backpatch(goto_after_bool_assign, after_bool_assign_label)
             return temp_for_bool
        else:
            raise ValueError(f"Cannot obtain a valid 'place' for {context_msg} from evaluation result: {eval_result}")


    def _parse_atom(self):
        """
        Parses the smallest units of an expression: literals, variables,
        parenthesized expressions, and function calls within expressions.
        Handles unary operators like '-' (negation) and 'όχι' (logical not).
        Returns:
            dict: The result of the parsed atom, containing either 'place' (for values)
                  or 'truelist'/'falselist' (for boolean results of 'όχι').
        """
        token = self.current_expr_token
        if not token:
            raise ValueError("Unexpected end of expression while parsing atom.")

        tok_type, tok_val = token[0], token[1]
        line_num = token[2] if len(token) >= 3 else '?'

        if tok_type in ['INTEGER', 'FLOAT']:
            self._advance_expr_token()
            return {'place': tok_val}

        elif tok_type == 'VARIABLE':
            var_name = tok_val
            is_embedded_func_call = False
            if self.expr_pos + 1 < len(self.expr_token_list):
                next_token = self.expr_token_list[self.expr_pos + 1]
                if next_token[1] == '(':
                    try:
                        entry = self.symbol_table.lookup(var_name, self.gen_body_scope_level)
                        if entry['entry_type'] == 'function':
                            is_embedded_func_call = True
                    except SymbolTableError:
                         pass # Treat as regular variable if not a function

            if is_embedded_func_call:
                func_name_token = self.current_expr_token
                self._advance_expr_token() # Consume function name
                self._advance_expr_token() # Consume '('

                func_entry = self.symbol_table.lookup(func_name_token[1], self.gen_body_scope_level)
                expected_params = func_entry.get('parameters', {})
                expected_param_list = sorted(expected_params.values(), key=lambda item: item['order'])

                evaluated_args_for_par_quads = []
                arg_idx = 0

                if not (self.current_expr_token and self.current_expr_token[1] == ')'): # Check if args exist
                    while True:
                        if arg_idx >= len(expected_param_list):
                            raise ValueError(f"Too many arguments for embedded function call to '{func_name_token[1]}' near line {line_num}.")

                        formal_param_info = expected_param_list[arg_idx]
                        formal_param_name = formal_param_info['name']
                        expected_arg_mode = formal_param_info.get('mode', 'CV')

                        if expected_arg_mode == 'REF':
                             raise ValueError(f"Cannot pass by reference (REF) to parameter '{formal_param_name}' of function '{func_name_token[1]}' when called inside an expression near line {line_num}.")
                        if self.current_expr_token and self.current_expr_token[0] == 'PARAMETER_PASS':
                            raise ValueError(f"Cannot use '%' for argument {arg_idx+1} ('{formal_param_name}') of function '{func_name_token[1]}' when called inside an expression near line {line_num}.")

                        arg_eval_result = self._parse_expr_prec(0) # Parse argument expression
                        arg_place = self._place_from_eval_result(arg_eval_result, f"argument {arg_idx+1} of {func_name_token[1]} call")
                        evaluated_args_for_par_quads.append({'place': arg_place, 'mode': 'CV'}) # Always CV for embedded calls
                        arg_idx += 1
                        if self.current_expr_token and self.current_expr_token[1] == ',':
                            self._advance_expr_token() # Consume comma
                        elif self.current_expr_token and self.current_expr_token[1] == ')':
                            break # End of arguments
                        else:
                             err_tok = self.current_expr_token[1] if self.current_expr_token else 'EOF'
                             raise ValueError(f"Expected ',' or ')' after argument in embedded call to '{func_name_token[1]}' near line {line_num}, found '{err_tok}'.")

                if arg_idx != len(expected_param_list):
                    raise ValueError(f"Incorrect number of arguments for embedded call to '{func_name_token[1]}' near line {line_num}. Expected {len(expected_param_list)}, got {arg_idx}.")

                self._advance_expr_token() # Consume ')'
                return_temp = self.new_temp(target_scope_level=self.gen_body_scope_level)
                for arg_data in evaluated_args_for_par_quads:
                    self.emit('par', arg_data['place'], 'CV', '_')
                self.emit('par', return_temp, 'RET', '_')
                self.emit('call', func_name_token[1], '_', '_')
                return {'place': return_temp}

            else: # Simple Variable
                try:
                    self.symbol_table.lookup(var_name, self.gen_body_scope_level)
                except SymbolTableError as e:
                    raise ValueError(f"Expression Error near line {line_num}: {e}")

                self._advance_expr_token() # Consume variable name
                return {'place': var_name}

        elif tok_val == '(' or tok_val == '[': # Parenthesized Expression
            opening_bracket = tok_val
            closing_bracket = ')' if opening_bracket == '(' else ']'
            self._advance_expr_token() # Consume opening bracket
            result = self._parse_expr_prec(0) # Parse inner expression
            if not (self.current_expr_token and self.current_expr_token[1] == closing_bracket):
                err_tok = self.current_expr_token[1] if self.current_expr_token else 'EOF'
                raise ValueError(f"Expected closing bracket '{closing_bracket}' near line {line_num}, but found '{err_tok}'.")
            self._advance_expr_token() # Consume closing bracket
            return result

        elif tok_val == '-' and tok_type == 'OPERATOR': # Unary Minus
            self._advance_expr_token()
            operand_result = self._parse_expr_prec(OP_PRECEDENCE['UMINUS'])
            operand_place = self._place_from_eval_result(operand_result, "operand of unary '-'")
            temp = self.new_temp()
            self.emit('-', 0, operand_place, temp)
            return {'place': temp}

        elif tok_val == '+' and tok_type == 'OPERATOR': # Unary Plus
            self._advance_expr_token()
            result = self._parse_expr_prec(OP_PRECEDENCE['UPLUS'])
            return result

        elif tok_val == 'όχι' and tok_type == 'KEYWORD': # Logical Not
            self._advance_expr_token()
            operand_result = self._parse_expr_prec(OP_PRECEDENCE['όχι'])
            boolean_operand = self._ensure_boolean(operand_result, 'όχι', line_num)
            return {'truelist': boolean_operand['falselist'], 'falselist': boolean_operand['truelist']}

        else:
            raise ValueError(f"Unexpected token starting expression atom: {token} near line {line_num}")

    def _parse_expr_prec(self, min_precedence):
        """
        Parses expression components using precedence climbing (a form of recursive descent).
        Handles binary operators based on their precedence relative to `min_precedence`.
        Args:
            min_precedence (int): The minimum precedence level an operator must have
                                  to be processed at this level of recursion.
        Returns:
            dict: The result of the parsed (sub)expression, containing either
                  'place' or 'truelist'/'falselist'.
        """
        left_result = self._parse_atom()

        while True:
            current_op_token = self.current_expr_token
            if not current_op_token: 
                break

            op_prec = self._get_op_precedence(current_op_token)
            if op_prec < min_precedence: 
                break

            op_val = current_op_token[1]
            op_line = current_op_token[2] if len(current_op_token) >= 3 else '?'

            if op_val == 'και':
                bool_left = self._ensure_boolean(left_result, 'και', op_line)
                self._advance_expr_token()
                right_eval_start_label = self.nextquad()
                self.backpatch(bool_left['truelist'], right_eval_start_label)
                right_result = self._parse_expr_prec(OP_PRECEDENCE['και'] + 1)
                bool_right = self._ensure_boolean(right_result, 'και', op_line)
                left_result = {
                    'truelist': bool_right['truelist'],
                    'falselist': self.mergelist(bool_left['falselist'], bool_right['falselist'])
                }
            elif op_val == 'ή':
                bool_left = self._ensure_boolean(left_result, 'ή', op_line)
                self._advance_expr_token()
                right_eval_start_label = self.nextquad()
                self.backpatch(bool_left['falselist'], right_eval_start_label)
                right_result = self._parse_expr_prec(OP_PRECEDENCE['ή'] + 1)
                bool_right = self._ensure_boolean(right_result, 'ή', op_line)
                left_result = {
                    'truelist': self.mergelist(bool_left['truelist'], bool_right['truelist']),
                    'falselist': bool_right['falselist']
                }
            else: # Arithmetic and Relational Operators
                self._advance_expr_token()
                right_result = self._parse_expr_prec(op_prec + 1)
                left_place = self._place_from_eval_result(left_result, f"left operand of '{op_val}'")
                right_place = self._place_from_eval_result(right_result, f"right operand of '{op_val}'")

                if op_val in ARITHMETIC_OPERATORS:
                    new_temp = self.new_temp()
                    self.emit(op_val, left_place, right_place, new_temp)
                    left_result = {'place': new_temp}
                elif op_val in RELATIONAL_OPERATORS:
                    true_list = self.makelist(self.emit(op_val, left_place, right_place, '?'))
                    false_list = self.makelist(self.emit('jump', '_', '_', '?'))
                    left_result = {'truelist': true_list, 'falselist': false_list}
                else:
                    raise ValueError(f"Unsupported or misplaced binary operator: '{op_val}' near line {op_line}")
        return left_result

    def evaluate_expression(self, expr_tokens_from_parser):
        """
        Evaluates an expression represented by a list of tokens, generating quads.
        Handles standalone function calls (detected by parser) or general expressions.
        Args:
            expr_tokens_from_parser (list): List of tokens or a list containing a
                                           single ('FUNCTION_CALL', ...) node.
        Returns:
            dict: The evaluation result {'place': ...} or {'truelist': ..., 'falselist': ...}.
        """
        if not expr_tokens_from_parser:
            raise ValueError("ICG: Cannot evaluate an empty expression token list.")

        if len(expr_tokens_from_parser) == 1 and \
           isinstance(expr_tokens_from_parser[0], tuple) and \
           len(expr_tokens_from_parser[0]) > 0 and \
           expr_tokens_from_parser[0][0] == 'FUNCTION_CALL': # Standalone Function Call
            _, func_name, args_info_list = expr_tokens_from_parser[0]
            evaluated_args_for_par = []
            for i, arg_info in enumerate(args_info_list):
                arg_expr_tokens = arg_info['expression']
                arg_eval_result = self.evaluate_expression(arg_expr_tokens) # Recursive call
                arg_place = self._place_from_eval_result(arg_eval_result, f"argument {i+1} for function call to '{func_name}'")
                arg_mode = arg_info['mode']
                evaluated_args_for_par.append({'place': arg_place, 'mode': arg_mode})

            ret_temp = self.new_temp(target_scope_level=self.gen_body_scope_level)
            for arg_data in evaluated_args_for_par:
                self.emit('par', arg_data['place'], arg_data['mode'], '_')
            self.emit('par', ret_temp, 'RET', '_')
            self.emit('call', func_name, '_', '_')
            return {'place': ret_temp}
        else: # General Expression
            self._init_expr_parser(expr_tokens_from_parser)
            result = self._parse_expr_prec(0)
            if self.current_expr_token is not None:
                pass
            return result

    # --- AST Node Generation Dispatch ---

    def generate(self, node):
        """
        Recursively generates intermediate code for a given AST node.
        Dispatches to specific handlers based on the AST node type.
        Args:
            node (tuple or None): The AST node to process.
        Returns:
            dict: A dictionary potentially containing 'nextlist' for control flow statements.
                  Returns {'nextlist': emptylist()} for simple statements.
        """
        if node is None: # Handles empty statements or optional blocks
             return {'nextlist': self.emptylist()}
        if not isinstance(node, tuple) or len(node) == 0:
             return {'nextlist': self.emptylist()} 

        node_type = node[0]
        caller_st_entry_context = self.current_generating_block_st_entry
        caller_gen_body_scope_level_context = self.gen_body_scope_level
        result_info = {'nextlist': self.emptylist()}

        if node_type == 'PROGRAM':
            _, name, _, subs_ast, main_stmts_ast = node
            program_st_entry = self.symbol_table.lookup(name, 0)
            self._set_gen_context(program_st_entry)

            main_jump_quad_label = None
            if subs_ast:
                main_jump_quad_label = self.emit('jump', '_', '_', '?') # Jump over subprograms

            for sub_node in subs_ast: # Generate code for subprograms first
                self.generate(sub_node)

            self._set_gen_context(program_st_entry) # Restore context for main program
            main_start_label = self.nextquad()
            if main_jump_quad_label:
                self.backpatch(self.makelist(main_jump_quad_label), main_start_label)

            self.emit('begin_block', name, '_', '_')
            main_body_nextlist = self._generate_statement_list(main_stmts_ast)
            self.backpatch(main_body_nextlist, self.nextquad()) # Patch exits to after halt
            self.emit('halt', '_', '_', '_')
            self.emit('end_block', name, '_', '_')

        elif node_type == 'FUNCTION' or node_type == 'PROCEDURE':
            _, name, _, _, stmts_ast, definition_level = node
            func_proc_st_entry = self.symbol_table.lookup(name, definition_level)
            self._set_gen_context(func_proc_st_entry)

            start_label = self.nextquad()
            self.symbol_table.update_entry(name, definition_level, start_quad=start_label)

            self.emit('begin_block', name, '_', '_')
            body_nextlist = self._generate_statement_list(stmts_ast)
            self.backpatch(body_nextlist, self.nextquad()) # Patch exits to after end_block
            self.emit('end_block', name, '_', '_')

        elif node_type == 'IF':
            _, cond_tokens, then_stmts, else_stmts = node
            cond_result = self.evaluate_expression(cond_tokens)
            line_num = cond_tokens[0][2] if cond_tokens and len(cond_tokens[0]) >=3 else '?'
            cond_bool = self._ensure_boolean(cond_result, 'εάν', line_num)

            then_start_label = self.nextquad()
            self.backpatch(cond_bool['truelist'], then_start_label)
            then_exits = self._generate_statement_list(then_stmts)

            if else_stmts:
                goto_after_else = self.makelist(self.emit('jump', '_', '_', '?'))
                else_start_label = self.nextquad()
                self.backpatch(cond_bool['falselist'], else_start_label)
                else_exits = self._generate_statement_list(else_stmts)
                result_info['nextlist'] = self.mergelist(then_exits, goto_after_else, else_exits)
            else:
                result_info['nextlist'] = self.mergelist(then_exits, cond_bool['falselist'])

        elif node_type == 'WHILE':
            _, cond_tokens, body_stmts = node
            loop_cond_start_label = self.nextquad()
            cond_result = self.evaluate_expression(cond_tokens)
            line_num = cond_tokens[0][2] if cond_tokens and len(cond_tokens[0]) >=3 else '?'
            cond_bool = self._ensure_boolean(cond_result, 'όσο', line_num)

            loop_body_start_label = self.nextquad()
            self.backpatch(cond_bool['truelist'], loop_body_start_label)

            body_exits = self._generate_statement_list(body_stmts)
            self.backpatch(body_exits, loop_cond_start_label) # Patch body exits back to condition
            self.emit('jump', '_', '_', loop_cond_start_label) # Jump from end of body to condition
            result_info['nextlist'] = cond_bool['falselist'] # Exits from loop are when condition is false

        elif node_type == 'REPEAT_UNTIL':
             _, body_stmts, cond_tokens = node
             loop_body_start_label = self.nextquad()
             body_exits = self._generate_statement_list(body_stmts)

             cond_eval_start_label = self.nextquad()
             self.backpatch(body_exits, cond_eval_start_label) # Patch body exits to condition check

             cond_result = self.evaluate_expression(cond_tokens)
             line_num = cond_tokens[0][2] if cond_tokens and len(cond_tokens[0]) >=3 else '?'
             cond_bool = self._ensure_boolean(cond_result, 'μέχρι', line_num)

             self.backpatch(cond_bool['falselist'], loop_body_start_label) # If false, repeat body
             result_info['nextlist'] = cond_bool['truelist'] # If true, exit loop

        elif node_type == 'FOR':
            _, var_name, start_toks, end_toks, step_toks, body_stmts = node
            start_eval = self.evaluate_expression(start_toks)
            start_place = self._place_from_eval_result(start_eval, "FOR loop start value")
            self.emit(':=', start_place, '_', var_name) # var := start

            loop_cond_label = self.nextquad() # Label for condition check
            end_eval = self.evaluate_expression(end_toks)
            end_place = self._place_from_eval_result(end_eval, "FOR loop end value")
            step_eval = self.evaluate_expression(step_toks)
            step_place = self._place_from_eval_result(step_eval, "FOR loop step value")

            # Condition: var <= end 
            cond_true_list = self.makelist(self.emit('<=', var_name, end_place, '?'))
            cond_false_list = self.makelist(self.emit('jump', '_', '_', '?')) # Exit loop

            loop_body_label = self.nextquad() # Label for loop body
            self.backpatch(cond_true_list, loop_body_label)
            body_exits = self._generate_statement_list(body_stmts)

            step_update_label = self.nextquad() # Label for step update
            self.backpatch(body_exits, step_update_label) # Patch body exits to step update
            self.emit('+', var_name, step_place, var_name) # var := var + step
            self.emit('jump', '_', '_', loop_cond_label) # Jump back to condition check
            result_info['nextlist'] = cond_false_list # Exits from loop when condition is false

        elif node_type == 'ASSIGNMENT':
            _, var_name, expr_tokens = node
            eval_result = self.evaluate_expression(expr_tokens)
            rhs_place = self._place_from_eval_result(eval_result, f"right side of assignment to '{var_name}'")
            self.emit(':=', rhs_place, '_', var_name)
            result_info['nextlist'] = self.emptylist()

        elif node_type == 'PRINT':
            _, expr_list_of_lists = node
            for expr_tokens in expr_list_of_lists:
                eval_result = self.evaluate_expression(expr_tokens)
                place_to_print = self._place_from_eval_result(eval_result, "value for 'γράψε'")
                self.emit('out', place_to_print, '_', '_')
            result_info['nextlist'] = self.emptylist()

        elif node_type == 'READ':
            _, var_list = node
            for var_name in var_list:
                self.emit('in', var_name, '_', '_')
            result_info['nextlist'] = self.emptylist()

        elif node_type == 'EXECUTE':
            _, proc_name, args_info_list = node
            for i, arg_info in enumerate(args_info_list):
                arg_expr_tokens = arg_info['expression']
                arg_eval_result = self.evaluate_expression(arg_expr_tokens)
                arg_place = self._place_from_eval_result(arg_eval_result, f"argument {i+1} for procedure '{proc_name}'")
                arg_mode = arg_info['mode']
                self.emit('par', arg_place, arg_mode, '_')
            self.emit('call', proc_name, '_', '_')
            result_info['nextlist'] = self.emptylist()

        self._restore_gen_context(caller_st_entry_context, caller_gen_body_scope_level_context)
        return result_info

    def _generate_statement_list(self, statements_ast_list):
        """
        Generates code for a list of statements, handling backpatching between them.
        Each statement's 'nextlist' (if any) is backpatched to the start of the next statement.
        Args:
            statements_ast_list (list): A list of statement AST nodes.
        Returns:
            list: The merged 'nextlist' containing all unresolved jumps out of this sequence.
        """
        current_overall_nextlist = self.emptylist()
        for i, stmt_node in enumerate(statements_ast_list):
            if stmt_node is None: 
                continue # Skip empty statements
            start_of_current_stmt_label = self.nextquad()
            self.backpatch(current_overall_nextlist, start_of_current_stmt_label)
            stmt_result = self.generate(stmt_node)
            current_overall_nextlist = stmt_result.get('nextlist', self.emptylist())
        return current_overall_nextlist

    def get_code(self):
        """
        Formats the generated quadruples into a list of human-readable strings.
        Returns:
            list: A list of strings, where each string represents a formatted quadruple.
        """
        lines = []
        if not self.quads: 
            return lines
        
        max_label_len = 0
        if self.quads:
            max_q_label_num = 0
            for q_check in self.quads:
                if q_check['label'] > max_q_label_num:
                    max_q_label_num = q_check['label']
            if max_q_label_num > 0 :
                 max_label_len = len(str(max_q_label_num))
            else: 
                max_label_len = 1 # Default if only one quad or no labels > 0

        for q in self.quads:
            label_str = str(q['label']).ljust(max_label_len)
            result_str = str(q['result']) 
            lines.append(f"{label_str} : {q['op']}, {q['arg1']}, {q['arg2']}, {result_str}")
        return lines
    
# --- Final Code Generator ---
class FinalCodeGenerator:
    """
    Generates final target code (RISC-V assembly) from the intermediate quadruples,
    using the Symbol Table for variable offsets, types, and scope information.
    """
    GREEK_CHAR_MAP = {
        'α': 'a', 'β': 'v', 'γ': 'g', 'δ': 'd', 'ε': 'e', 'ζ': 'z', 'η': 'i', 'θ': 'th',
        'ι': 'i', 'κ': 'k', 'λ': 'l', 'μ': 'm', 'ν': 'n', 'ξ': 'x', 'ο': 'o', 'π': 'p',
        'ρ': 'r', 'σ': 's', 'ς': 's', 'τ': 't', 'υ': 'y', 'φ': 'f', 'χ': 'ch', 'ψ': 'ps', 'ω': 'o',
        'ά': 'a', 'έ': 'e', 'ή': 'i', 'ί': 'i', 'ό': 'o', 'ύ': 'y', 'ώ': 'o',
        'ϊ': 'i', 'ϋ': 'y', 'ΐ': 'i', 'ΰ': 'y',
        'Α': 'A', 'Β': 'V', 'Γ': 'G', 'Δ': 'D', 'Ε': 'E', 'Ζ': 'Z', 'Η': 'I', 'Θ': 'Th',
        'Ι': 'I', 'Κ': 'K', 'Λ': 'L', 'Μ': 'M', 'Ν': 'N', 'Ξ': 'X', 'Ο': 'O', 'Π': 'P',
        'Ρ': 'R', 'Σ': 'S', 'Τ': 'T', 'Υ': 'Y', 'Φ': 'F', 'Χ': 'Ch', 'Ψ': 'Ps', 'Ω': 'O',
        'Ά': 'A', 'Έ': 'E', 'Ή': 'I', 'Ί': 'I', 'Ό': 'O', 'Ύ': 'Y', 'Ώ': 'O', 'Ϊ': 'I', 'Ϋ': 'Y',
    }
    
    def __init__(self, quads, symbol_table, program_name):
        """
        Initializes the Final Code Generator.
        Args:
            quads (list): The list of intermediate code quadruples.
            symbol_table (SymbolTable): The symbol table.
            program_name (str): The name of the main program block.
        """
        self.quads = quads
        self.symbol_table = symbol_table
        self.program_name = program_name
        self.riscv_code = []
        self.asm_labels = {}
        self.func_entry_asm_labels = {}

        self.current_processing_block_st_entry = None
        self.current_processing_block_def_level = -1
        self.current_processing_block_body_level = -1
        self.param_passing_stack_offset_for_current_call = 0

        self.SP = "sp"
        self.RA = "ra"
        self.S0_FP = "s0"
        self.GP = "gp"
        self.T0, self.T1, self.T2, self.T3 = "t0", "t1", "t2", "t3"
        self.A0 = "a0"
        self.A7 = "a7"

        self.SAVED_OLD_FP_OFFSET = 0
        self.SAVED_RA_ON_STACK_OFFSET = 4
        self.ACCESS_LINK_ON_STACK_OFFSET = 8
        self.RET_VAL_ADDR_ON_STACK_OFFSET = 12
        self.FIRST_PARAM_BASE_OFFSET_FUNCTION = 16
        self.FIRST_PARAM_BASE_OFFSET_PROCEDURE = 12
        self.ST_DATA_AREA_START_OFFSET = -self.symbol_table.ST_OFFSET_BASE

        self._initialize_labels_and_funcs()

    def _add_instr(self, instruction, comment=None, indent=True):
        """Adds a formatted RISC-V assembly instruction to the code list."""
        prefix = "    " if indent else ""
        if comment:
            self.riscv_code.append(f"{prefix}{instruction.ljust(35)} # {comment}")
        else:
            self.riscv_code.append(f"{prefix}{instruction}")

    def _add_label(self, label_name):
        """Adds an assembly label to the code list."""
        self.riscv_code.append(f"{label_name}:")
    
    def _sanitize_label_name(self, name):
        """
        Converts a name into a valid ASCII
        assembly label.
        """
        if not name: 
            return "unnamed_label"
        name_str = str(name)
        sanitized_parts = []
        for char_original in name_str:
            if char_original in FinalCodeGenerator.GREEK_CHAR_MAP:
                transliterated = FinalCodeGenerator.GREEK_CHAR_MAP[char_original]
                sanitized_parts.append(transliterated)
            elif ('a' <= char_original.lower() <= 'z') or \
                 ('0' <= char_original <= '9') or \
                 (char_original == '_'):
                sanitized_parts.append(char_original)
            elif char_original == '-':
                sanitized_parts.append('_')
            else:
                sanitized_parts.append('_') # Fallback for any other character
        result = "".join(sanitized_parts)
        if not result: 
            return "sanitized_empty_label"
        if result and result[0].isdigit(): #label doesn't start with a digit
            result = "L_" + result
        return result.lower() # lowercase for consistency
    
    def _initialize_labels_and_funcs(self):
        """ Pre-calculates assembly labels for all quad labels and block entry points. """
        for quad in self.quads:
            self.asm_labels[quad['label']] = f"LQ{quad['label']}"
            if quad['op'] == 'begin_block':
                original_block_name = quad['arg1']
                sanitized_block_name = self._sanitize_label_name(original_block_name)
                #naming convention for block entries
                self.func_entry_asm_labels[original_block_name] = f"L_{sanitized_block_name}_entry"
        
    def _get_st_entry_for_final_gen(self, name, current_block_definition_level):
        """
        Looks up a symbol entry. Prioritizes parameters of the current block,
        then uses standard symbol table lookup.
        """
        # Case 1: Accessing a parameter of the current function/procedure
        if self.current_processing_block_st_entry and \
           self.current_processing_block_st_entry['entry_type'] in ['function', 'procedure']:
            params_dict = self.current_processing_block_st_entry.get('parameters', {})
            if name in params_dict:
                param_entry_details = params_dict[name]
                full_param_entry = {
                    'name': name,
                    'entry_type': 'parameter', 
                    'mode': param_entry_details.get('mode'),
                    'order': param_entry_details.get('order'),
                    'offset': param_entry_details.get('offset'), 
                    'data_type': param_entry_details.get('type', 'integer'),
                    'size': 4, 
                    'scope_level': self.current_processing_block_body_level 
                }
                return full_param_entry

        # Case 2: Not a parameter of current func/proc, or in main program block.
        lookup_start_scope_level = -1
        if self.current_processing_block_st_entry:
            block_type = self.current_processing_block_st_entry.get('entry_type', 'None')
            if block_type == 'program':
                 lookup_start_scope_level = 0
            else: # func/proc, if it was not in its params_dict
                 lookup_start_scope_level = self.current_processing_block_body_level
        else:
             lookup_start_scope_level = 0 # Fallback to global if no context
        if lookup_start_scope_level < 0: 
            lookup_start_scope_level = 0 

        try:
            found_entry = self.symbol_table.lookup(name, lookup_start_scope_level)
            entry_type = found_entry.get('entry_type')
            # Check for offset only if it's a data-holding type that's NOT a function/proc definition
            if entry_type in ['variable', 'temp_variable'] and 'offset' not in found_entry:
                 print(f"CRITICAL WARNING: FCG lookup for '{name}' (type {entry_type}) returned entry without 'offset': {found_entry}", file=sys.stderr)
                 raise SymbolTableError(f"Internal FCG Error: Entry for data symbol '{name}' lacks 'offset'.")
            return found_entry
        except SymbolTableError as e:
             current_block_name = self.current_processing_block_st_entry.get('name', 'None') if self.current_processing_block_st_entry else 'None'
             raise SymbolTableError(f"FinalCodeGen Error: Symbol '{name}' not found. Lookup started from scope level {lookup_start_scope_level} (Current block: {current_block_name}). Original error: {e}")

    def _get_block_data_allocation_size(self, block_st_entry):
        """
        Calculates the total stack space needed for parameters, locals, and temps
        within a specific block's activation record data area, ensuring alignment.
        """
        block_def_level = block_st_entry['scope_level']
        scope_level_of_data_area = -1

        if block_st_entry['entry_type'] == 'program':
            scope_level_of_data_area = 0 # Main program's data 
        else: # Function/Procedure
            scope_level_of_data_area = block_def_level + 1 

        if scope_level_of_data_area < 0:
             print(f"Warning: Cannot determine data area scope level for block '{block_st_entry.get('name', 'N/A')}'.")
             return 0

        # Retrieve the final offset 
        final_offset_in_scope = self.symbol_table.frame_offsets.get(scope_level_of_data_area, self.symbol_table.ST_OFFSET_BASE)

        data_area_size = final_offset_in_scope - self.symbol_table.ST_OFFSET_BASE
        data_area_size = max(0, data_area_size) # Ensure non-negative

        # Ensure 4-byte alignment 
        if data_area_size % 4 != 0:
            aligned_size = ((data_area_size // 4) + 1) * 4
            return aligned_size
        else:
            return data_area_size

    def _gnvlcode(self, var_name_to_find_addr, target_addr_reg):
        """
        Generates code to find the address of a non-local variable by traversing
        the static chain (access links). The address is stored in target_addr_reg.
        """
        var_entry = self._get_st_entry_for_final_gen(var_name_to_find_addr, self.current_processing_block_def_level)
        var_scope_level = var_entry['scope_level']
        var_containing_block_def_level = var_scope_level - 1 if var_scope_level > 0 else 0 
        levels_to_go_up = self.current_processing_block_def_level - var_containing_block_def_level

        if levels_to_go_up < 0:
             raise SymbolTableError(f"Internal Error (_gnvlcode): Cannot access variable '{var_name_to_find_addr}' defined at block level {var_containing_block_def_level} from block level {self.current_processing_block_def_level}.")

        # Start with the current frame pointer
        self._add_instr(f"mv {target_addr_reg}, {self.S0_FP}", f"gnvl: start FP for {var_name_to_find_addr}")

        for i in range(levels_to_go_up):
            self._add_instr(f"lw {target_addr_reg}, {self.ACCESS_LINK_ON_STACK_OFFSET}({target_addr_reg})", f"follow Access Link #{i+1}")

  
        # Calculate final address using the variable's offset.
        var_st_offset = var_entry['offset']
        if var_st_offset is None: 
            KeyError(f"_gnvlcode: Offset missing for non-local '{var_name_to_find_addr}'")
        self._add_instr(f"addi {target_addr_reg}, {target_addr_reg}, {-var_st_offset}", f"final addr of {var_name_to_find_addr} (-{var_st_offset} from its block's FP)")

    def _loadvr(self, operand_name_or_literal, target_val_reg):
        """
        Generates code to load the value of an operand (literal or variable)
        into a target register. Handles various addressing modes.
        """
        op_str = str(operand_name_or_literal)

        # --- Case 1: Literal Value ---
        if op_str.isdigit() or (op_str.startswith('-') and op_str[1:].isdigit()):
            self._add_instr(f"li {target_val_reg}, {op_str}", f"load literal value {op_str}")
            return

        # --- Case 2: Variable Name ---
        var_entry = self._get_st_entry_for_final_gen(op_str, self.current_processing_block_def_level)
        var_st_offset = var_entry.get('offset')
        var_scope_level = var_entry['scope_level']
        is_ref_param = var_entry.get('entry_type') == 'parameter' and var_entry.get('mode') == 'REF'
        addr_reg = self.T3 

        is_global = (var_scope_level == 0)
        is_param_of_current_block = (
            self.current_processing_block_st_entry and
            self.current_processing_block_st_entry['entry_type'] != 'program' and 
            var_entry.get('entry_type') == 'parameter' and
            var_scope_level == self.current_processing_block_body_level
        )
        is_local_or_temp_of_current_block = (
             self.current_processing_block_st_entry and
             self.current_processing_block_st_entry['entry_type'] != 'program' and 
             var_entry.get('entry_type') in ['variable', 'temp_variable'] and
             var_scope_level == self.current_processing_block_body_level
        )
        is_func_return_var = ( 
            var_entry.get('is_return_var', False) and
            self.current_processing_block_st_entry and
            self.current_processing_block_st_entry['entry_type'] == 'function' and
            var_entry['name'] == self.current_processing_block_st_entry.get('name') and
            var_scope_level == self.current_processing_block_body_level
        )
        if is_func_return_var: 
            is_local_or_temp_of_current_block = False # Prevent double match

        # --- Generate code to get the variable's ADDRESS into addr_reg (T3) ---
        if is_param_of_current_block:
            param_order_index = var_entry.get('order', 0)
            is_function = self.current_processing_block_st_entry.get('entry_type') == 'function'
            base_offset_on_stack = self.FIRST_PARAM_BASE_OFFSET_FUNCTION if is_function else self.FIRST_PARAM_BASE_OFFSET_PROCEDURE
            formal_params_dict = self.current_processing_block_st_entry.get('parameters', {})
            num_formal_params = len(formal_params_dict)
            if num_formal_params == 0: 
                 current_block_name_for_error = self.current_processing_block_st_entry.get('name', 'UNKNOWN_BLOCK')
                 raise ValueError(f"FCG _loadvr Error: Parameter '{op_str}' identified, but current block '{current_block_name_for_error}' has no parameters defined in its symbol table entry.")
            # Parameters are pushed in order, so last param is at lowest positive offset from FP base for params.
            effective_stack_order_from_base = (num_formal_params - 1 - param_order_index)
            param_actual_positive_offset = base_offset_on_stack + (effective_stack_order_from_base * 4)
            self._add_instr(f"addi {addr_reg}, {self.S0_FP}, {param_actual_positive_offset}", f"addr of param {op_str} (ord {param_order_index}, eff_stk_ord {effective_stack_order_from_base}) (+{param_actual_positive_offset} from s0)")
        elif is_local_or_temp_of_current_block:
            if var_st_offset is None: 
                raise KeyError(f"FCG _loadvr: Offset missing for local/temp '{op_str}'")
            self._add_instr(f"addi {addr_reg}, {self.S0_FP}, {-var_st_offset}", f"addr of local/temp {op_str} (-{var_st_offset} from s0)")
        elif is_global:
            if var_st_offset is None: 
                raise KeyError(f"FCG _loadvr: Offset missing for global '{op_str}'")
            self._add_instr(f"addi {addr_reg}, {self.GP}, {-var_st_offset}", f"addr of global {op_str} (-{var_st_offset} from gp)")
        elif is_func_return_var: # Reading from function's own name 
            print(f"Warning: Reading from function name '{op_str}' on RHS is unusual.")
            if var_st_offset is None: 
                raise KeyError(f"FCG _loadvr: Offset missing for func return var '{op_str}'")
            self._add_instr(f"addi {addr_reg}, {self.S0_FP}, {-var_st_offset}", f"addr of func ret var {op_str} (-{var_st_offset} from s0)")
        else: # Non-local variable
            self._gnvlcode(op_str, addr_reg) 

        if is_ref_param:
             # Parameter is REF: Load the address stored in the parameter slot, then load value using that address
             self._add_instr(f"lw {addr_reg}, 0({addr_reg})", f"deref 1: get actual addr from REF param {op_str}")
             self._add_instr(f"lw {target_val_reg}, 0({addr_reg})", f"deref 2: load value using actual addr for {op_str}")
        else:
             # Parameter is CV or Variable/Temp: load value directly from calculated address
             self._add_instr(f"lw {target_val_reg}, 0({addr_reg})", f"load value for {op_str}")

    def _storerv(self, source_val_reg, target_var_name):
        """
        Generates code to store the value from a source register into a target
        variable. Handles function return value assignment and various addressing modes.
        """
        var_entry = self._get_st_entry_for_final_gen(target_var_name, self.current_processing_block_def_level)
        addr_reg = self.T3 

        # Check if assigning to the current function's own name (for return value)
        is_func_return_var_assignment = False
        if (var_entry.get('entry_type') == 'function' or var_entry.get('is_return_var') is True) and \
           self.current_processing_block_st_entry and \
           self.current_processing_block_st_entry['entry_type'] == 'function' and \
           var_entry['name'] == self.current_processing_block_st_entry.get('name'):
            is_func_return_var_assignment = True

        if is_func_return_var_assignment:
            # Get the return value ADDRESS into addr_reg
            self._add_instr(f"lw {addr_reg}, {self.RET_VAL_ADDR_ON_STACK_OFFSET}({self.S0_FP})", f"get return value addr for {target_var_name}")
            # Store the value from source_val_reg into the location pointed to by addr_reg
            self._add_instr(f"sw {source_val_reg}, 0({addr_reg})", f"store {target_var_name} return value")
            return 
        try:
            var_st_offset = var_entry['offset']
        except KeyError:
            print(f"ERROR in _storerv: 'offset' key missing in entry for '{target_var_name}'. Entry was: {var_entry}", file=sys.stderr)
            raise

        var_scope_level = var_entry['scope_level']
        is_ref_param = var_entry.get('entry_type') == 'parameter' and var_entry.get('mode') == 'REF'
        is_global = (var_scope_level == 0)
        is_param_of_current_block = (
            self.current_processing_block_st_entry and
            self.current_processing_block_st_entry['entry_type'] != 'program' and
            var_entry.get('entry_type') == 'parameter' and
            var_scope_level == self.current_processing_block_body_level
        )
        is_local_or_temp_of_current_block = (
             self.current_processing_block_st_entry and
             self.current_processing_block_st_entry['entry_type'] != 'program' and
             var_entry.get('entry_type') in ['variable', 'temp_variable'] and
             var_scope_level == self.current_processing_block_body_level
        )

        # --- Generate code to get the variable's ADDRESS into addr_reg (T3) ---
        if is_param_of_current_block:
            param_order_index = var_entry.get('order', 0)
            is_function = self.current_processing_block_st_entry.get('entry_type') == 'function'
            base_offset_on_stack = self.FIRST_PARAM_BASE_OFFSET_FUNCTION if is_function else self.FIRST_PARAM_BASE_OFFSET_PROCEDURE
            formal_params_dict = self.current_processing_block_st_entry.get('parameters', {})
            num_formal_params = len(formal_params_dict)
            if num_formal_params == 0:
                 current_block_name_for_error = self.current_processing_block_st_entry.get('name', 'UNKNOWN_BLOCK')
                 raise ValueError(f"FCG _storerv Error: Parameter '{target_var_name}' identified, but current block '{current_block_name_for_error}' has no parameters defined in its symbol table entry.")
            effective_stack_order_from_base = (num_formal_params - 1 - param_order_index)
            param_actual_positive_offset = base_offset_on_stack + (effective_stack_order_from_base * 4)
            self._add_instr(f"addi {addr_reg}, {self.S0_FP}, {param_actual_positive_offset}", f"addr of param {target_var_name} (ord {param_order_index}, eff_stk_ord {effective_stack_order_from_base}) (+{param_actual_positive_offset} from s0)")
        elif is_local_or_temp_of_current_block:
            self._add_instr(f"addi {addr_reg}, {self.S0_FP}, {-var_st_offset}", f"addr of local/temp {target_var_name} (-{var_st_offset} from s0)")
        elif is_global:
            self._add_instr(f"addi {addr_reg}, {self.GP}, {-var_st_offset}", f"addr of global {target_var_name} (-{var_st_offset} from gp)")
        else: # Non-local variable
            self._gnvlcode(target_var_name, addr_reg) # Puts address into addr_reg

        # --- Store the VALUE from source_val_reg to the calculated address ---
        if is_ref_param: # If the target variable itself is a REF parameter
             # Load the actual target address from the parameter slot
             self._add_instr(f"lw {addr_reg}, 0({addr_reg})", f"get final addr from REF param {target_var_name}")
             # Store using the final address
             self._add_instr(f"sw {source_val_reg}, 0({addr_reg})", f"store to actual addr pointed by {target_var_name}")
        else:
             # For CV params, locals, temps, globals: store directly to calculated address
             self._add_instr(f"sw {source_val_reg}, 0({addr_reg})", f"store value to {target_var_name}")

    def _translate_quad(self, quad_idx):
        """ Translates a single quadruple into RISC-V assembly instructions. """
        quad = self.quads[quad_idx]
        op, arg1, arg2, result = quad['op'], quad['arg1'], quad['arg2'], quad['result']

        self._add_label(self.asm_labels[quad['label']])
        self._add_instr(f"# Quad {quad['label']}: {op}, {arg1}, {arg2}, {result}", comment=None, indent=False)

        # --- Dispatch based on Quad Operator ---

        if op == 'begin_block':
            block_name = arg1
            # Find the symbol table entry for this block
            parent_def_level_for_lookup = -1 # Default for looking up main program
            if self.current_processing_block_st_entry and self.current_processing_block_st_entry['entry_type'] != 'program':
                 parent_def_level_for_lookup = self.current_processing_block_def_level

            lookup_scope = 0 if block_name == self.program_name else max(0, parent_def_level_for_lookup)
            try:
                 self.current_processing_block_st_entry = self.symbol_table.lookup(block_name, lookup_scope)
            except SymbolTableError as e:
                 raise RuntimeError(f"FCG Error: Could not find ST entry for begin_block '{block_name}' (looked up from scope {lookup_scope}): {e}")

            # Update FCG context trackers
            self.current_processing_block_def_level = self.current_processing_block_st_entry['scope_level']
            if self.current_processing_block_st_entry['entry_type'] == 'program':
                self.current_processing_block_body_level = 0
            else:
                self.current_processing_block_body_level = self.current_processing_block_def_level + 1

            data_items_allocation_size = self._get_block_data_allocation_size(self.current_processing_block_st_entry)
            self._add_label(self.func_entry_asm_labels[block_name])

            # --- Generate Block Prologue ---
            if block_name == self.program_name:
                if data_items_allocation_size > 0:
                    self._add_instr(f"addi {self.SP}, {self.SP}, -{data_items_allocation_size}", f"allocate frame for main ({data_items_allocation_size} bytes)")
                self._add_instr(f"mv {self.GP}, {self.SP}", "set Global Pointer (GP = main frame base)")
                self._add_instr(f"mv {self.S0_FP}, {self.SP}", "set Frame Pointer S0 for main")
            else: # Function/Procedure Prologue
                self._add_instr(f"addi {self.SP}, {self.SP}, -{self.SAVED_RA_ON_STACK_OFFSET}", "make space for RA")
                self._add_instr(f"sw {self.RA}, 0({self.SP})", "save RA")
                self._add_instr(f"addi {self.SP}, {self.SP}, -4", "make space for old S0/FP")
                self._add_instr(f"sw {self.S0_FP}, 0({self.SP})", "save caller's S0/FP")
                self._add_instr(f"mv {self.S0_FP}, {self.SP}", "set new Frame Pointer S0")
                if data_items_allocation_size > 0:
                    self._add_instr(f"addi {self.SP}, {self.SP}, -{data_items_allocation_size}", f"allocate locals/temps for {block_name} ({data_items_allocation_size} bytes)")

        elif op == 'end_block':
            block_name = arg1
            try:
                 block_entry = self.symbol_table.lookup(block_name, 0 if block_name == self.program_name else self.current_processing_block_def_level)
                 data_items_allocation_size = self._get_block_data_allocation_size(block_entry)
            except SymbolTableError:
                 print(f"Warning: FCG could not find block entry '{block_name}' during end_block. Using default size 0.")
                 data_items_allocation_size = 0

            # --- Generate Block Epilogue ---
            if block_name == self.program_name:
                 if data_items_allocation_size > 0:
                     self._add_instr(f"addi {self.SP}, {self.SP}, {data_items_allocation_size}", f"deallocate main's frame ({data_items_allocation_size} bytes)")
            else: 
                self._add_instr(f"mv {self.SP}, {self.S0_FP}", "deallocate locals/temps (SP = S0)")
                self._add_instr(f"lw {self.S0_FP}, {self.SAVED_OLD_FP_OFFSET}({self.SP})", "restore caller's S0/FP")
                self._add_instr(f"lw {self.RA}, {self.SAVED_RA_ON_STACK_OFFSET}({self.SP})", "restore RA")
                self._add_instr(f"addi {self.SP}, {self.SP}, 8", "pop old S0/FP and RA")
                self._add_instr(f"jr {self.RA}", f"return from {block_name}")

        elif op == ':=':
            self._loadvr(arg1, self.T0) # Load value of arg1 into T0
            self._storerv(self.T0, result) # Store value from T0 into result variable

        elif op in ARITHMETIC_OPERATORS: # '+', '-', '*', '/'
            self._loadvr(arg1, self.T0) # Load first operand into T0
            self._loadvr(arg2, self.T1) # Load second operand into T1
            op_map = {'+': 'add', '-': 'sub', '*': 'mul', '/': 'div'} # Integer operations
            self._add_instr(f"{op_map[op]} {self.T2}, {self.T0}, {self.T1}", f"{result} = {arg1} {op} {arg2}") # Perform operation, result in T2
            self._storerv(self.T2, result) # Store result from T2 into result variable

        elif op in RELATIONAL_OPERATORS: 
            self._loadvr(arg1, self.T0) 
            self._loadvr(arg2, self.T1) 
            branch_op_map = {
                '=': 'beq', '==': 'beq', '<>': 'bne',
                '<': 'blt', '<=': 'ble', '>': 'bgt', '>=': 'bge', '<-': 'blt'
            }
            try:
                target_quad_label = int(result)
                target_asm_label = self.asm_labels[target_quad_label]
                self._add_instr(f"{branch_op_map[op]} {self.T0}, {self.T1}, {target_asm_label}", f"if {arg1} {op} {arg2} goto {target_asm_label}")
            except (ValueError, KeyError):
                 self._add_instr(f"# ERROR: Invalid jump target '{result}' for op '{op}'", comment="Compilation Error", indent=False)

        elif op == 'jump':
            try:
                target_quad_label = int(result)
                target_asm_label = self.asm_labels[target_quad_label]
                self._add_instr(f"j {target_asm_label}", f"goto {target_asm_label}")
            except (ValueError, KeyError):
                 self._add_instr(f"# ERROR: Invalid jump target '{result}' for op 'jump'", comment="Compilation Error", indent=False)

        elif op == 'par':
            param_val_operand = arg1
            param_mode = arg2
            callers_base_reg = self.GP if self.current_processing_block_st_entry['entry_type'] == 'program' else self.S0_FP

            if param_mode == 'CV':
                self._loadvr(param_val_operand, self.T0) # Get value into T0
                self._add_instr(f"addi {self.SP}, {self.SP}, -4", "make space on stack for CV param")
                self._add_instr(f"sw {self.T0}, 0({self.SP})", f"push CV value of '{param_val_operand}'")
                self.param_passing_stack_offset_for_current_call += 4
            elif param_mode == 'REF':
                var_entry_ref = self._get_st_entry_for_final_gen(param_val_operand, self.current_processing_block_def_level)
                addr_to_pass_reg = self.T0 # Register to hold the final address to push
                arg_var_scope_level = var_entry_ref['scope_level']
                arg_var_st_offset = var_entry_ref.get('offset')
                arg_is_ref_param_of_caller = (var_entry_ref.get('entry_type') == 'parameter' and var_entry_ref.get('mode') == 'REF')

                if arg_var_st_offset is None and arg_var_scope_level > 0: # Check offset needed for non-globals
                    raise KeyError(f"FCG par REF: Offset missing for argument '{param_val_operand}'")

                if arg_var_scope_level == 0: # Argument is global
                    self._add_instr(f"addi {addr_to_pass_reg}, {self.GP}, {-arg_var_st_offset}", f"addr of global arg '{param_val_operand}' for REF")
                elif var_entry_ref.get('entry_type') == 'parameter' and arg_var_scope_level == self.current_processing_block_body_level: # Argument is a parameter of the caller function/proc
                    caller_param_order = var_entry_ref['order']
                    caller_is_func = self.current_processing_block_st_entry['entry_type'] == 'function'
                    caller_base_offset = self.FIRST_PARAM_BASE_OFFSET_FUNCTION if caller_is_func else self.FIRST_PARAM_BASE_OFFSET_PROCEDURE
                    num_formal_params_caller = len(self.current_processing_block_st_entry.get('parameters', {}))
                    eff_stack_ord_caller_param = (num_formal_params_caller - 1 - caller_param_order)
                    caller_param_actual_pos_offset = caller_base_offset + (eff_stack_ord_caller_param * 4)
                    self._add_instr(f"addi {addr_to_pass_reg}, {callers_base_reg}, {caller_param_actual_pos_offset}", f"addr of caller's param slot for '{param_val_operand}'")
                    if arg_is_ref_param_of_caller: 
                         self._add_instr(f"lw {addr_to_pass_reg}, 0({addr_to_pass_reg})", f"get actual addr from caller's REF param '{param_val_operand}'")
                elif var_entry_ref.get('entry_type') in ['variable', 'temp_variable'] and arg_var_scope_level == self.current_processing_block_body_level: # Argument is a local variable of the caller
                     self._add_instr(f"addi {addr_to_pass_reg}, {callers_base_reg}, {-arg_var_st_offset}", f"addr of local arg '{param_val_operand}' for REF")
                else: # Argument is non-local to the caller (defined in an enclosing scope)
                     self._gnvlcode(param_val_operand, addr_to_pass_reg) # Get address 
                     if arg_is_ref_param_of_caller:
                          self._add_instr(f"lw {addr_to_pass_reg}, 0({addr_to_pass_reg})", f"get actual addr from non-local REF var '{param_val_operand}'")

                self._add_instr(f"addi {self.SP}, {self.SP}, -4", "make space for REF param addr")
                self._add_instr(f"sw {addr_to_pass_reg}, 0({self.SP})", f"push REF addr for '{param_val_operand}'")
                self.param_passing_stack_offset_for_current_call += 4
            elif param_mode == 'RET':
                ret_temp_name = param_val_operand
                temp_entry_for_ret = self._get_st_entry_for_final_gen(ret_temp_name, self.current_processing_block_def_level)
                ret_temp_offset = temp_entry_for_ret.get('offset')
                if ret_temp_offset is None: 
                    raise KeyError(f"FCG par RET: Offset missing for RET temp '{ret_temp_name}'")
                self._add_instr(f"addi {self.T0}, {callers_base_reg}, {-ret_temp_offset}", f"addr of RET temp '{ret_temp_name}' in caller")
                self._add_instr(f"addi {self.SP}, {self.SP}, -4", "make space for RET_VAL_ADDR")
                self._add_instr(f"sw {self.T0}, 0({self.SP})", "push RET_VAL_ADDR")
                self.param_passing_stack_offset_for_current_call += 4

        elif op == 'call':
            callee_name = arg1
            try:
                 callee_st_entry = self.symbol_table.lookup(callee_name, self.current_processing_block_def_level)
                 callee_def_level = callee_st_entry['scope_level']
            except SymbolTableError as e:
                  raise RuntimeError(f"FCG Error: Cannot find ST entry for called function/proc '{callee_name}'. {e}")

            caller_def_level = self.current_processing_block_def_level
            caller_fp_reg = self.S0_FP # Default for calls from within functions/procs
            if self.current_processing_block_st_entry['entry_type'] == 'program':
                 caller_fp_reg = self.GP # Main program uses GP as its 'frame pointer' context

            # --- Calculate and Push Access Link (Static Link) ---
            access_link_val_reg = self.T0 # Use T0 to compute the access link value
            if callee_def_level == caller_def_level + 1: 
                self._add_instr(f"mv {access_link_val_reg}, {caller_fp_reg}", "AL = Caller's FP (callee is child)")
            elif callee_def_level == caller_def_level: # Callee is at same nesting level 
                 if self.current_processing_block_st_entry['entry_type'] == 'program': # Main program calls a top-level function
                      self._add_instr(f"mv {access_link_val_reg}, {self.GP}", "AL = Main's GP (main calls top-level)")
                 else: # Sibling call (e.g., function A calls function B, both defined in main)
                      self._add_instr(f"lw {access_link_val_reg}, {self.ACCESS_LINK_ON_STACK_OFFSET}({caller_fp_reg})", "AL = Caller's AL (callee is sibling)")
            elif caller_def_level > callee_def_level: # Callee is in an outer scope
                 self._add_instr(f"mv {access_link_val_reg}, {caller_fp_reg}", "AL: start traversal from caller's FP")
                 levels_to_ascend = caller_def_level - callee_def_level
                 for _ in range(levels_to_ascend):
                     self._add_instr(f"lw {access_link_val_reg}, {self.ACCESS_LINK_ON_STACK_OFFSET}({access_link_val_reg})", "AL: follow static link up")
            else: 
                 raise RuntimeError(f"FCG Error: Invalid nesting relationship for Access Link calculation. Call from '{self.current_processing_block_st_entry.get('name', 'N/A')}' (lvl {caller_def_level}) to '{callee_name}' (lvl {callee_def_level}).")

            self._add_instr(f"addi {self.SP}, {self.SP}, -4", "make space for Access Link")
            self._add_instr(f"sw {access_link_val_reg}, 0({self.SP})", "push Access Link")
            self.param_passing_stack_offset_for_current_call += 4

            # --- Perform the Call ---
            self._add_instr(f"jal {self.func_entry_asm_labels[callee_name]}", f"call {callee_name}")

            # --- Clean up Stack After Call Returns ---
            if self.param_passing_stack_offset_for_current_call > 0:
                self._add_instr(f"addi {self.SP}, {self.SP}, {self.param_passing_stack_offset_for_current_call}", f"pop params/ret_addr/AL ({self.param_passing_stack_offset_for_current_call} bytes)")
            self.param_passing_stack_offset_for_current_call = 0 

        elif op == 'in': 
            target_var_name = arg1
            self._add_instr(f"li {self.A7}, 5", "syscall: read_integer")
            self._add_instr("ecall") # Input value is returned in A0
            self._storerv(self.A0, target_var_name)

        elif op == 'out': 
            value_to_print_operand = arg1
            self._loadvr(value_to_print_operand, self.A0)
            self._add_instr(f"li {self.A7}, 1", "syscall: print_integer")
            self._add_instr("ecall")
            self._add_instr(f"li {self.A0}, 10", "load newline char value") 
            self._add_instr(f"li {self.A7}, 11", "syscall: print_character")
            self._add_instr("ecall")

        elif op == 'halt':
            self._add_instr(f"li {self.A0}, 0", "exit code 0")
            self._add_instr(f"li {self.A7}, 93", "syscall: exit")
            self._add_instr("ecall")
        else:
             self._add_instr(f"# UNHANDLED QUAD: {op}, {arg1}, {arg2}, {result}", comment="Warning", indent=False)


    def generate(self):
        """
        Generates the complete RISC-V assembly code by translating all quadruples.
        Includes boilerplate for data/text sections and initial jump.
        """
        self.riscv_code.append(".data")

        self.riscv_code.append("\n.text")
        self.riscv_code.append(".globl main") # Standard entry point label for linker

        # --- Initial Jump ---
        # skipping over function/procedure definitions.
        if self.quads:
            first_quad = self.quads[0]
            if first_quad['op'] == 'jump' and first_quad['arg1'] == '_' and first_quad['arg2'] == '_':
                # This is the expected jump from ICG to skip subprogram code
                try:
                    target_main_label_quad_num = int(first_quad['result'])
                    if target_main_label_quad_num in self.asm_labels:
                         self._add_label("main") # Standard entry point
                         self._add_instr(f"j {self.asm_labels[target_main_label_quad_num]}", "Initial jump to main code")
                    else: # Should not happen if ICG is correct
                         print(f"Warning: FCG initial jump target '{first_quad['result']}' invalid. Starting at first quad label.")
                         self._add_label("main")
                         self._add_instr(f"j {self.asm_labels[first_quad['label']]}", "Fallback: jump to first quad label")
                except (ValueError, KeyError): # If result is not a valid label num
                     print(f"Warning: FCG cannot parse initial jump target '{first_quad['result']}'. Starting at first quad label.")
                     self._add_label("main")
                     self._add_instr(f"j {self.asm_labels[first_quad['label']]}", "Fallback: jump to first quad label")
            elif self.program_name in self.func_entry_asm_labels:
                 # Fallback if ICG didn't start with a jump (e.g., program has no subprograms)
                 self._add_label("main")
                 self._add_instr(f"j {self.func_entry_asm_labels[self.program_name]}", f"Initial jump to {self.program_name} entry")
            else:
                  # Further fallback if main program entry label isn't found (should be an error)
                  print("Warning: FCG could not determine main program entry point. Starting at first quad label.")
                  self._add_label("main")
                  self._add_instr(f"j {self.asm_labels[first_quad['label']]}", "Fallback: jump to first quad label")
        else: # No quads, just exit
             self._add_label("main")
             self._add_instr("li a7, 93", "syscall: exit (no code)")
             self._add_instr("ecall")

        context_stack = [] 
        for i in range(len(self.quads)):
            quad = self.quads[i]
            if quad['op'] == 'begin_block':
                context_stack.append({
                    'st_entry': self.current_processing_block_st_entry,
                    'def_level': self.current_processing_block_def_level,
                    'body_level': self.current_processing_block_body_level
                })

            self._translate_quad(i)

            if quad['op'] == 'end_block':
                if context_stack:
                    prev_context = context_stack.pop()
                    self.current_processing_block_st_entry = prev_context['st_entry']
                    self.current_processing_block_def_level = prev_context['def_level']
                    self.current_processing_block_body_level = prev_context['body_level']
                

        return "\n".join(self.riscv_code)

# --- Main Function ---
def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <source_file.gr>")
        sys.exit(1)

    source_file = sys.argv[1]
    base_name = source_file.rsplit('.', 1)[0] if '.' in source_file else source_file
    intermediate_file = base_name + ".int"
    symbol_table_file = base_name + ".sym"
    final_code_file = base_name + ".asm"

    try:
        with open(source_file, "r", encoding="utf-8") as f:
            code = f.read()
    except IOError as e:
        print(f"ERROR: Cannot open source file '{source_file}': {e}", file=sys.stderr)
        sys.exit(1)

    tokens = lexer(code)
    mismatch_tokens = [t for t in tokens if t[0] == 'MISMATCH']
    if mismatch_tokens:
        print("\nLEXICAL ERRORS FOUND:", file=sys.stderr)
        for token_type, msg, line in mismatch_tokens:
            print(f"  Line {line}: {msg}", file=sys.stderr)
        print("Compilation aborted.", file=sys.stderr)
        sys.exit(1)
    ast = None
    parser_instance = None

    try:
        with open(symbol_table_file, "w", encoding="utf-8") as sym_f:
            print(f"Symbol table output will be written to: {symbol_table_file}")
            parser_instance = GreekPlusPlusParser(tokens, sym_file_handle=sym_f)
            ast = parser_instance.parse()

            sym_f.write("\n\n--- Symbol Table State AFTER PARSING ---\n")
            for i, scope_dict in enumerate(parser_instance.symbol_table.scopes):
                sym_f.write(f"--- Scope Level {i} (Parser View) ---\n")
                sorted_items = sorted(scope_dict.items(), key=lambda item: item[1].get('offset', float('inf')))
                formatted_scope = pprint.pformat(dict(sorted_items), indent=2, width=100)
                sym_f.write(formatted_scope + "\n")
            sym_f.write("\n--- Frame Offset Tracking AFTER PARSING ---\n")
            sym_f.write(pprint.pformat(parser_instance.symbol_table.frame_offsets, indent=2, width=100) + "\n")
    except (SyntaxError, SymbolTableError):
        print("Compilation aborted due to parsing/semantic errors.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print("\nINTERNAL PARSING ERROR:", file=sys.stderr)
        print(e, file=sys.stderr)
        print("Compilation aborted.", file=sys.stderr)
        sys.exit(1)

    if not ast or not parser_instance:
        print("ERROR: Parsing failed to produce an AST. Aborting.", file=sys.stderr)
        sys.exit(1)

    icg_instance = IntermediateCodeGenerator(parser_instance.symbol_table)
    intermediate_quads_text = []
    try:
        icg_instance.generate(ast)
        intermediate_quads_text = icg_instance.get_code()

        with open(intermediate_file, "w", encoding="utf-8") as f_int:
            for line in intermediate_quads_text:
                f_int.write(line + "\n")
        print(f"Intermediate code written to: {intermediate_file}")

        with open(symbol_table_file, "a", encoding="utf-8") as sym_f:
            sym_f.write("\n\n--- Symbol Table State AFTER ICG (includes temporaries) ---\n")
            st_after_icg = parser_instance.symbol_table
            for i, scope_dict in enumerate(st_after_icg.scopes):
                sym_f.write(f"--- Scope Level {i} (ICG View) ---\n")
                sorted_items = sorted(scope_dict.items(), key=lambda item: item[1].get('offset', float('inf')))
                formatted_scope = pprint.pformat(dict(sorted_items), indent=2, width=100)
                sym_f.write(formatted_scope + "\n")
            sym_f.write("\n--- Frame Offset Tracking AFTER ICG ---\n")
            sym_f.write(pprint.pformat(st_after_icg.frame_offsets, indent=2, width=100) + "\n")

    except (ValueError, SymbolTableError) as e:
        print(f"\nINTERMEDIATE CODE GENERATION ERROR: {e}", file=sys.stderr)
        print("Compilation aborted.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print("\nINTERNAL ICG ERROR:", file=sys.stderr)
        print(e, file=sys.stderr)
        print("Compilation aborted.", file=sys.stderr)
        sys.exit(1)

    if not icg_instance.quads:
        print("Warning: No intermediate code generated (AST might be empty or only contained declarations). Skipping final code generation.", file=sys.stderr)
    
    program_name_from_ast = ast[1] if ast[0] == 'PROGRAM' else "UnknownProgram"
    final_code_gen = FinalCodeGenerator(icg_instance.quads, parser_instance.symbol_table, program_name_from_ast)
    riscv_assembly_code = ""
    try:
        riscv_assembly_code = final_code_gen.generate()
        with open(final_code_file, "w", encoding="utf-8") as f_final:
            f_final.write(riscv_assembly_code)
        print(f"Final RISC-V code written to: {final_code_file}")

    except (SymbolTableError, RuntimeError, ValueError, KeyError) as e:
        print(f"\nFINAL CODE GENERATION ERROR: {e}", file=sys.stderr)
        if riscv_assembly_code:
            print("\n--- Partially Generated RISC-V Code (Error Occurred) ---", file=sys.stderr)
            print(riscv_assembly_code, file=sys.stderr)
            print("------------------------------------------------------", file=sys.stderr)
        print("Compilation aborted.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print("\nINTERNAL FCG ERROR:", file=sys.stderr)
        print(e, file=sys.stderr)
        print("Compilation aborted.", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()