#!/usr/bin/env python3
"""
PhiASM - Hybrydowy Interpreter i REPL (Ulepszona wersja)
Nowoczesny język niskiego poziomu z modularnym VM, bezpiecznymi makrami, enum flagami i lepszą obsługą etykiet.
"""

import re
import sys
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass
from enum import Enum, auto
import pyreadline3   # dla lepszego REPL

# --- Kolory instrukcji (do wyświetlania, nie w kodzie źródłowym) ---
class InstructionColor(Enum):
    MEMORY = "blue"
    ARITHMETIC = "red"
    CONTROL = "green"
    CONDITION = "yellow"
    IO = "cyan"
    MACRO = "magenta"

# --- Flagi jako Enum ---
class Flag(Enum):
    ZERO = auto()
    NEG = auto()
    CARRY = auto()

@dataclass
class Token:
    type: str
    value: str
    color: Optional[InstructionColor] = None
    line: int = 0
    column: int = 0

@dataclass
class Instruction:
    color: InstructionColor
    opcode: str
    operands: List[str]
    label: Optional[str] = None
    comment: Optional[str] = None
    line: int = 0
    label_addr: Optional[int] = None  # Adres docelowy dla skoków

class PhiASMTokenizer:
    """Tokenizer dla języka PhiASM (bez emoji w kodzie źródłowym)"""

    def __init__(self):
        self.tokens = []
        self.current_line = 0

    def tokenize(self, code: str) -> List[Token]:
        self.tokens = []
        lines = code.split('\n')
        for line_num, line in enumerate(lines, 1):
            self.current_line = line_num
            self._tokenize_line(line.strip())
        return self.tokens

    def _tokenize_line(self, line: str):
        if not line or line.startswith(';'):
            return
        comment_pos = line.find(';')
        if comment_pos != -1:
            comment = line[comment_pos+1:].strip()
            line = line[:comment_pos].strip()
        else:
            comment = None
        # Etykieta
        if ':' in line and not any(op in line for op in ['MOV', 'ADD', 'SUB', 'JMP', 'CMP', 'OUT', 'IN']):
            label = line.replace(':', '').strip()
            self.tokens.append(Token('LABEL', label, line=self.current_line))
            return
        # Podziel na części
        parts = line.split()
        if not parts:
            return
        opcode = parts[0]
        operands = []
        if len(parts) > 1:
            operand_str = ' '.join(parts[1:])
            operands = [op.strip() for op in operand_str.split(',')]
        self.tokens.append(Token('INSTRUCTION', opcode, line=self.current_line))
        for operand in operands:
            if operand:
                self.tokens.append(Token('OPERAND', operand, line=self.current_line))
        if comment:
            self.tokens.append(Token('COMMENT', comment, line=self.current_line))

class PhiASMParser:
    """Parser tworzący AST z tokenów i rozwiązuje etykiety"""

    def __init__(self):
        self.instructions = []
        self.labels = {}

    def parse(self, tokens: List[Token]) -> List[Instruction]:
        self.instructions = []
        self.labels = {}
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token.type == 'LABEL':
                self.labels[token.value] = len(self.instructions)
                i += 1
            elif token.type == 'INSTRUCTION':
                instruction = self._parse_instruction(tokens, i)
                if instruction:
                    self.instructions.append(instruction)
                    i += self._count_instruction_tokens(tokens, i)
                else:
                    i += 1
            else:
                i += 1
        # Rozwiąż adresy etykiet dla instrukcji skoku
        for instr in self.instructions:
            if instr.opcode.upper() in {"JMP", "JMPZ", "JZ", "JNZ", "CALL"} and instr.operands:
                label = instr.operands[0]
                if label in self.labels:
                    instr.label_addr = self.labels[label]
        return self.instructions

    def _parse_instruction(self, tokens: List[Token], start: int) -> Optional[Instruction]:
        if start >= len(tokens):
            return None
        instr_token = tokens[start]
        if instr_token.type != 'INSTRUCTION':
            return None
        operands = []
        i = start + 1
        while i < len(tokens) and tokens[i].type == 'OPERAND':
            operands.append(tokens[i].value)
            i += 1
        return Instruction(
            color=InstructionColor.MEMORY,  # Kolor przypisywany w REPL/UI
            opcode=instr_token.value,
            operands=operands,
            line=instr_token.line
        )

    def _count_instruction_tokens(self, tokens: List[Token], start: int) -> int:
        count = 1
        i = start + 1
        while i < len(tokens) and tokens[i].type in ['OPERAND', 'COMMENT']:
            count += 1
            i += 1
        return count

class PhiASMVirtualMachine:
    """Maszyna wirtualna dla PhiASM z modularnym dispatch i enum flagami"""

    def __init__(self, num_registers=16, memory_size=65536, input_func: Optional[Callable[[str], str]] = None):
        self.registers = [0] * num_registers
        self.memory = [0] * memory_size
        self.stack = []
        self.pc = 0
        self.flags = set()  # Set[Flag]
        self.running = True
        self.output_buffer = []
        self.input_buffer = []
        self.breakpoints = set()
        self.input_func = input_func or input
        self.instruction_set = self._build_instruction_set()

    def reset(self):
        self.registers = [0] * len(self.registers)
        self.memory = [0] * len(self.memory)
        self.stack = []
        self.pc = 0
        self.flags = set()
        self.running = True
        self.output_buffer = []

    def set_register(self, reg: str, value: int):
        idx = self._get_register_index(reg)
        if idx is not None:
            self.registers[idx] = value & 0xFFFFFFFF

    def get_register(self, reg: str) -> int:
        idx = self._get_register_index(reg)
        return self.registers[idx] if idx is not None else 0

    def _get_register_index(self, reg: str) -> Optional[int]:
        if reg.upper().startswith('R'):
            try:
                reg_num = int(reg[1:])
                if 0 <= reg_num < len(self.registers):
                    return reg_num
            except ValueError:
                pass
        return None

    def parse_operand(self, operand: str) -> int:
        operand = operand.strip()
        # Hex literal
        if operand.startswith('0x') or operand.startswith('0X'):
            return int(operand, 16)
        # Binary literal
        if operand.startswith('0b') or operand.startswith('0B'):
            return int(operand, 2)
        # Decimal literal
        if operand.isdigit() or (operand.startswith('-') and operand[1:].isdigit()):
            return int(operand)
        # Register
        if operand.upper().startswith('R'):
            return self.get_register(operand)
        # Memory address
        if operand.startswith('[') and operand.endswith(']'):
            addr = self.parse_operand(operand[1:-1])
            if 0 <= addr < len(self.memory):
                return self.memory[addr]
        return 0

    def _build_instruction_set(self):
        return {
            'MOV': self._instr_mov,
            'ADD': self._instr_add,
            'SUB': self._instr_sub,
            'MUL': self._instr_mul,
            'DIV': self._instr_div,
            'MOD': self._instr_mod,
            'AND': self._instr_and,
            'OR': self._instr_or,
            'XOR': self._instr_xor,
            'NOT': self._instr_not,
            'JMP': self._instr_jmp,
            'JMPZ': self._instr_jmpz,
            'JZ': self._instr_jmpz,
            'JNZ': self._instr_jmpnz,
            'CALL': self._instr_call,
            'RET': self._instr_ret,
            'PUSH': self._instr_push,
            'POP': self._instr_pop,
            'SHL': self._instr_shl,
            'SHR': self._instr_shr,
            'ROL': self._instr_rol,
            'ROR': self._instr_ror,
            'CMP': self._instr_cmp,
            'OUT': self._instr_out,
            'IN': self._instr_in,
            'HALT': self._instr_halt,
            'END': self._instr_halt,
            'PRINT': self._instr_print,
        }

    def execute_instruction(self, instruction: Instruction) -> bool:
        opcode = instruction.opcode.upper()
        handler = self.instruction_set.get(opcode)
        if self.pc in self.breakpoints:
            print(f"Breakpoint hit at PC={self.pc}, instruction: {opcode}")
            return False
        try:
            if handler:
                return handler(instruction)
            else:
                print(f"Nieznana instrukcja: {opcode}")
        except Exception as e:
            print(f"Błąd wykonania instrukcji {opcode}: {e}")
            return False
        self.pc += 1
        return True

    # --- Instrukcje ---
    def _instr_mov(self, instr: Instruction):
        dest, src = instr.operands[0], instr.operands[1]
        value = self.parse_operand(src)
        self.set_register(dest, value)
        self.pc += 1
        return True

    def _instr_add(self, instr: Instruction):
        dest = instr.operands[0]
        if len(instr.operands) == 3:
            val1 = self.parse_operand(instr.operands[1])
            val2 = self.parse_operand(instr.operands[2])
        else:
            val1 = self.get_register(dest)
            val2 = self.parse_operand(instr.operands[1])
        result = val1 + val2
        self.set_register(dest, result)
        self._update_flags(result)
        self.pc += 1
        return True

    def _instr_sub(self, instr: Instruction):
        dest = instr.operands[0]
        if len(instr.operands) == 3:
            val1 = self.parse_operand(instr.operands[1])
            val2 = self.parse_operand(instr.operands[2])
        else:
            val1 = self.get_register(dest)
            val2 = self.parse_operand(instr.operands[1])
        result = val1 - val2
        self.set_register(dest, result)
        self._update_flags(result)
        self.pc += 1
        return True

    def _instr_mul(self, instr: Instruction):
        dest = instr.operands[0]
        val1 = self.parse_operand(instr.operands[1])
        val2 = self.parse_operand(instr.operands[2])
        result = val1 * val2
        self.set_register(dest, result)
        self._update_flags(result)
        self.pc += 1
        return True

    def _instr_div(self, instr: Instruction):
        dest = instr.operands[0]
        val1 = self.parse_operand(instr.operands[1])
        val2 = self.parse_operand(instr.operands[2])
        result = val1 // val2 if val2 != 0 else 0
        self.set_register(dest, result)
        self._update_flags(result)
        self.pc += 1
        return True

    def _instr_mod(self, instr: Instruction):
        dest = instr.operands[0]
        val1 = self.parse_operand(instr.operands[1])
        val2 = self.parse_operand(instr.operands[2])
        result = val1 % val2 if val2 != 0 else 0
        self.set_register(dest, result)
        self._update_flags(result)
        self.pc += 1
        return True

    def _instr_and(self, instr: Instruction):
        dest = instr.operands[0]
        val1 = self.parse_operand(instr.operands[1])
        val2 = self.parse_operand(instr.operands[2])
        result = val1 & val2
        self.set_register(dest, result)
        self._update_flags(result)
        self.pc += 1
        return True

    def _instr_or(self, instr: Instruction):
        dest = instr.operands[0]
        val1 = self.parse_operand(instr.operands[1])
        val2 = self.parse_operand(instr.operands[2])
        result = val1 | val2
        self.set_register(dest, result)
        self._update_flags(result)
        self.pc += 1
        return True

    def _instr_xor(self, instr: Instruction):
        dest = instr.operands[0]
        val1 = self.parse_operand(instr.operands[1])
        val2 = self.parse_operand(instr.operands[2])
        result = val1 ^ val2
        self.set_register(dest, result)
        self._update_flags(result)
        self.pc += 1
        return True

    def _instr_not(self, instr: Instruction):
        dest = instr.operands[0]
        val = self.parse_operand(instr.operands[1])
        result = ~val
        self.set_register(dest, result)
        self._update_flags(result)
        self.pc += 1
        return True

    def _instr_jmp(self, instr: Instruction):
        if instr.label_addr is not None:
            self.pc = instr.label_addr
        else:
            print(f"Nieznana etykieta: {instr.operands[0]}")
            self.pc += 1
        return True

    def _instr_jmpz(self, instr: Instruction):
        cond = self.get_register(instr.operands[1]) if len(instr.operands) > 1 else None
        if (Flag.ZERO in self.flags) or (cond is not None and cond == 0):
            if instr.label_addr is not None:
                self.pc = instr.label_addr
            else:
                print(f"Nieznana etykieta: {instr.operands[0]}")
                self.pc += 1
        else:
            self.pc += 1
        return True

    def _instr_jmpnz(self, instr: Instruction):
        cond = self.get_register(instr.operands[1]) if len(instr.operands) > 1 else None
        if (Flag.ZERO not in self.flags) and (cond is None or cond != 0):
            if instr.label_addr is not None:
                self.pc = instr.label_addr
            else:
                print(f"Nieznana etykieta: {instr.operands[0]}")
                self.pc += 1
        else:
            self.pc += 1
        return True

    def _instr_call(self, instr: Instruction):
        self.stack.append(self.pc + 1)
        if instr.label_addr is not None:
            self.pc = instr.label_addr
        else:
            print(f"Nieznana etykieta: {instr.operands[0]}")
            self.pc += 1
        return True

    def _instr_ret(self, instr: Instruction):
        if self.stack:
            self.pc = self.stack.pop()
        else:
            print("Stos pusty przy RET")
            self.pc += 1
        return True

    def _instr_push(self, instr: Instruction):
        value = self.parse_operand(instr.operands[0])
        self.stack.append(value)
        self.pc += 1
        return True

    def _instr_pop(self, instr: Instruction):
        if self.stack:
            value = self.stack.pop()
            self.set_register(instr.operands[0], value)
        self.pc += 1
        return True

    def _instr_shl(self, instr: Instruction):
        dest = instr.operands[0]
        val = self.parse_operand(instr.operands[1])
        shift = self.parse_operand(instr.operands[2])
        result = val << shift
        self.set_register(dest, result)
        self._update_flags(result)
        self.pc += 1
        return True

    def _instr_shr(self, instr: Instruction):
        dest = instr.operands[0]
        val = self.parse_operand(instr.operands[1])
        shift = self.parse_operand(instr.operands[2])
        result = val >> shift
        self.set_register(dest, result)
        self._update_flags(result)
        self.pc += 1
        return True

    def _instr_rol(self, instr: Instruction):
        dest = instr.operands[0]
        val = self.parse_operand(instr.operands[1])
        shift = self.parse_operand(instr.operands[2]) % 32
        result = ((val << shift) | (val >> (32 - shift))) & 0xFFFFFFFF
        self.set_register(dest, result)
        self._update_flags(result)
        self.pc += 1
        return True

    def _instr_ror(self, instr: Instruction):
        dest = instr.operands[0]
        val = self.parse_operand(instr.operands[1])
        shift = self.parse_operand(instr.operands[2]) % 32
        result = ((val >> shift) | (val << (32 - shift))) & 0xFFFFFFFF
        self.set_register(dest, result)
        self._update_flags(result)
        self.pc += 1
        return True

    def _instr_cmp(self, instr: Instruction):
        val1 = self.parse_operand(instr.operands[0])
        val2 = self.parse_operand(instr.operands[1])
        result = val1 - val2
        self._update_flags(result)
        self.pc += 1
        return True

    def _instr_out(self, instr: Instruction):
        value = self.parse_operand(instr.operands[0])
        self.output_buffer.append(str(value))
        print(f">> [WYJŚCIE]: {value}")
        self.pc += 1
        return True

    def _instr_in(self, instr: Instruction):
        dest = instr.operands[0]
        if self.input_buffer:
            value = self.input_buffer.pop(0)
        else:
            try:
                value = int(self.input_func(">> [WEJŚCIE]: "))
            except ValueError:
                value = 0
        self.set_register(dest, value)
        self.pc += 1
        return True

    def _instr_halt(self, instr: Instruction):
        self.running = False
        return False

    def _instr_print(self, instr: Instruction):
        value = self.parse_operand(instr.operands[0])
        print(f"Wartość {instr.operands[0]} = {value}")
        self.pc += 1
        return True

    def _update_flags(self, result: int):
        self.flags.clear()
        if result == 0:
            self.flags.add(Flag.ZERO)
        if result < 0:
            self.flags.add(Flag.NEG)
        if result > 0xFFFFFFFF or result < 0:
            self.flags.add(Flag.CARRY)

class MacroProcessor:
    """Bezpieczny procesor makr z regexem"""

    def __init__(self):
        self.macros = {}

    def define_macro(self, name: str, body: str):
        self.macros[name] = body

    def expand_macros(self, code: str) -> str:
        for macro_name, macro_body in self.macros.items():
            pattern = rf'@{re.escape(macro_name)}\b'
            code = re.sub(pattern, macro_body, code)
        return code

class PhiASMInterpreter:
    """Główny interpreter PhiASM"""

    def __init__(self):
        self.tokenizer = PhiASMTokenizer()
        self.parser = PhiASMParser()
        self.vm = PhiASMVirtualMachine()
        self.macro_processor = MacroProcessor()
        self.debug_mode = False
        self.step_mode = False

    def load_file(self, filename: str) -> str:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()

    def execute_code(self, code: str):
        expanded_code = self.macro_processor.expand_macros(code)
        tokens = self.tokenizer.tokenize(expanded_code)
        instructions = self.parser.parse(tokens)
        self.vm.pc = 0
        self.vm.running = True
        while self.vm.running and self.vm.pc < len(instructions):
            instruction = instructions[self.vm.pc]
            if self.debug_mode:
                self._print_debug_info(instruction)
            if self.step_mode:
                input("Naciśnij Enter aby kontynuować...")
            if not self.vm.execute_instruction(instruction):
                break

    def _print_debug_info(self, instruction: Instruction):
        print(f"PC={self.vm.pc}: {instruction.opcode} {', '.join(instruction.operands)}")
        print(f"   Rejestry: R0={self.vm.registers[0]}, R1={self.vm.registers[1]}, R2={self.vm.registers[2]}")
        print(f"   Flagi: {[flag.name for flag in self.vm.flags]}")

    def add_breakpoint(self, line: int):
        self.vm.breakpoints.add(line)

    def show_registers(self):
        print("Stan rejestrów:")
        for i, val in enumerate(self.vm.registers[:8]):
            print(f"   R{i} = {val} (0x{val:08X})")

    def show_stack(self):
        print(f"Stos ({len(self.vm.stack)} elementów):")
        for i, val in enumerate(reversed(self.vm.stack[-5:])):
            print(f"   [{len(self.vm.stack)-1-i}] = {val}")

class PhiASMREPL:
    """REPL dla PhiASM"""

    def __init__(self):
        self.interpreter = PhiASMInterpreter()
        self.history = []
        self.commands = {
            'help': self._cmd_help,
            'reset': self._cmd_reset,
            'registers': self._cmd_registers,
            'stack': self._cmd_stack,
            'debug': self._cmd_debug,
            'step': self._cmd_step,
            'break': self._cmd_break,
            'run': self._cmd_run,
            'load': self._cmd_load,
            'macro': self._cmd_macro,
            'quit': self._cmd_quit,
            'exit': self._cmd_quit
        }

    def run(self):
        print("PhiASM REPL v2.0")
        print("Wpisz '.help' aby zobaczyć dostępne komendy")
        while True:
            try:
                line = input("PhiCode > ").strip()
                if not line:
                    continue
                self.history.append(line)
                if line.startswith('.'):
                    self._handle_command(line[1:])
                else:
                    try:
                        self.interpreter.execute_code(line)
                    except Exception as e:
                        print(f"Błąd: {e}")
            except KeyboardInterrupt:
                print("\nDo widzenia!")
                break
            except EOFError:
                break

    def _handle_command(self, command: str):
        parts = command.split()
        cmd = parts[0] if parts else ""
        args = parts[1:] if len(parts) > 1 else []
        if cmd in self.commands:
            self.commands[cmd](args)
        else:
            print(f"Nieznana komenda: {cmd}")

    def _cmd_help(self, args):
        print("""
Dostępne komendy:
  .help          - wyświetl tę pomoc
  .reset         - resetuj maszynę wirtualną
  .registers     - pokaż stan rejestrów
  .stack         - pokaż stan stosu
  .debug         - włącz/wyłącz tryb debug
  .step          - włącz/wyłącz tryb krok-po-kroku
  .break <line>  - ustaw breakpoint
  .run <file>    - uruchom plik
  .load <file>   - załaduj plik
  .macro <name> <body> - zdefiniuj makro
  .quit/.exit    - wyjdź z REPL
""")

    def _cmd_reset(self, args):
        self.interpreter.vm.reset()
        print("Maszyna wirtualna zresetowana")

    def _cmd_registers(self, args):
        self.interpreter.show_registers()

    def _cmd_stack(self, args):
        self.interpreter.show_stack()

    def _cmd_debug(self, args):
        self.interpreter.debug_mode = not self.interpreter.debug_mode
        state = "włączony" if self.interpreter.debug_mode else "wyłączony"
        print(f"Tryb debug {state}")

    def _cmd_step(self, args):
        self.interpreter.step_mode = not self.interpreter.step_mode
        state = "włączony" if self.interpreter.step_mode else "wyłączony"
        print(f"Tryb krok-po-kroku {state}")

    def _cmd_break(self, args):
        if args:
            try:
                line = int(args[0])
                self.interpreter.add_breakpoint(line)
                print(f"Breakpoint ustawiony na linii {line}")
            except ValueError:
                print("Podaj numer linii")
        else:
            print("Użycie: .break <numer_linii>")

    def _cmd_run(self, args):
        if args:
            try:
                code = self.interpreter.load_file(args[0])
                self.interpreter.execute_code(code)
            except Exception as e:
                print(f"Błąd: {e}")
        else:
            print("Użycie: .run <nazwa_pliku>")

    def _cmd_load(self, args):
        if args:
            try:
                code = self.interpreter.load_file(args[0])
                print(f"Załadowano plik: {args[0]}")
                print(f"Zawartość:\n{code}")
            except Exception as e:
                print(f"Błąd: {e}")
        else:
            print("Użycie: .load <nazwa_pliku>")

    def _cmd_macro(self, args):
        if len(args) >= 2:
            name = args[0]
            body = ' '.join(args[1:])
            self.interpreter.macro_processor.define_macro(name, body)
            print(f"Makro '{name}' zdefiniowane")
        else:
            print("Użycie: .macro <nazwa> <ciało_makra>")

    def _cmd_quit(self, args):
        print("Do widzenia!")
        sys.exit(0)

def main():
    if len(sys.argv) > 1:
        interpreter = PhiASMInterpreter()
        try:
            code = interpreter.load_file(sys.argv[1])
            interpreter.execute_code(code)
        except Exception as e:
            print(f"Błąd: {e}")
    else:
        repl = PhiASMREPL()
        repl.run()

if __name__ == "__main__":
    main()