.data

.text
.globl main
main:
    j LQ12                              # Initial jump to main code
LQ1:
# Quad 1: jump, _, _, 12
    j LQ12                              # goto LQ12
LQ2:
# Quad 2: begin_block, αύξηση, _, _
L_ayxisi_entry:
    addi sp, sp, -4                     # make space for RA
    sw ra, 0(sp)                        # save RA
    addi sp, sp, -4                     # make space for old S0/FP
    sw s0, 0(sp)                        # save caller's S0/FP
    mv s0, sp                           # set new Frame Pointer S0
    addi sp, sp, -16                    # allocate locals/temps for αύξηση (16 bytes)
LQ3:
# Quad 3: +, α, 1, t@1
    addi t3, s0, 20                     # addr of param α (ord 0, eff_stk_ord 1) (+20 from s0)
    lw t0, 0(t3)                        # load value for α
    li t1, 1                            # load literal value 1
    add t2, t0, t1                      # t@1 = α + 1
    addi t3, s0, -16                    # addr of local/temp t@1 (-16 from s0)
    sw t2, 0(t3)                        # store value to t@1
LQ4:
# Quad 4: :=, t@1, _, β
    addi t3, s0, -16                    # addr of local/temp t@1 (-16 from s0)
    lw t0, 0(t3)                        # load value for t@1
    addi t3, s0, 16                     # addr of param β (ord 1, eff_stk_ord 0) (+16 from s0)
    lw t3, 0(t3)                        # get final addr from REF param β
    sw t0, 0(t3)                        # store to actual addr pointed by β
LQ5:
# Quad 5: +, α, 1, t@2
    addi t3, s0, 20                     # addr of param α (ord 0, eff_stk_ord 1) (+20 from s0)
    lw t0, 0(t3)                        # load value for α
    li t1, 1                            # load literal value 1
    add t2, t0, t1                      # t@2 = α + 1
    addi t3, s0, -20                    # addr of local/temp t@2 (-20 from s0)
    sw t2, 0(t3)                        # store value to t@2
LQ6:
# Quad 6: :=, t@2, _, αύξηση
    addi t3, s0, -20                    # addr of local/temp t@2 (-20 from s0)
    lw t0, 0(t3)                        # load value for t@2
    lw t3, 12(s0)                       # get return value addr for αύξηση
    sw t0, 0(t3)                        # store αύξηση return value
LQ7:
# Quad 7: end_block, αύξηση, _, _
    mv sp, s0                           # deallocate locals/temps (SP = S0)
    lw s0, 0(sp)                        # restore caller's S0/FP
    lw ra, 4(sp)                        # restore RA
    addi sp, sp, 8                      # pop old S0/FP and RA
    jr ra                               # return from αύξηση
LQ8:
# Quad 8: begin_block, τύπωσε_συν_1, _, _
L_typose_syn_1_entry:
    addi sp, sp, -4                     # make space for RA
    sw ra, 0(sp)                        # save RA
    addi sp, sp, -4                     # make space for old S0/FP
    sw s0, 0(sp)                        # save caller's S0/FP
    mv s0, sp                           # set new Frame Pointer S0
    addi sp, sp, -16                    # allocate locals/temps for τύπωσε_συν_1 (16 bytes)
LQ9:
# Quad 9: +, χ, 1, t@3
    addi t3, s0, 12                     # addr of param χ (ord 0, eff_stk_ord 0) (+12 from s0)
    lw t0, 0(t3)                        # load value for χ
    li t1, 1                            # load literal value 1
    add t2, t0, t1                      # t@3 = χ + 1
    addi t3, s0, -24                    # addr of local/temp t@3 (-24 from s0)
    sw t2, 0(t3)                        # store value to t@3
LQ10:
# Quad 10: out, t@3, _, _
    addi t3, s0, -24                    # addr of local/temp t@3 (-24 from s0)
    lw a0, 0(t3)                        # load value for t@3
    li a7, 1                            # syscall: print_integer
    ecall
    li a0, 10                           # load newline char value
    li a7, 11                           # syscall: print_character
    ecall
LQ11:
# Quad 11: end_block, τύπωσε_συν_1, _, _
    mv sp, s0                           # deallocate locals/temps (SP = S0)
    lw s0, 0(sp)                        # restore caller's S0/FP
    lw ra, 4(sp)                        # restore RA
    addi sp, sp, 8                      # pop old S0/FP and RA
    jr ra                               # return from τύπωσε_συν_1
LQ12:
# Quad 12: begin_block, τεστ, _, _
L_test_entry:
    addi sp, sp, -48                    # allocate frame for main (48 bytes)
    mv gp, sp                           # set Global Pointer (GP = main frame base)
    mv s0, sp                           # set Frame Pointer S0 for main
LQ13:
# Quad 13: :=, 1, _, α
    li t0, 1                            # load literal value 1
    addi t3, gp, -12                    # addr of global α (-12 from gp)
    sw t0, 0(t3)                        # store value to α
LQ14:
# Quad 14: *, α, α, t@4
    addi t3, gp, -12                    # addr of global α (-12 from gp)
    lw t0, 0(t3)                        # load value for α
    addi t3, gp, -12                    # addr of global α (-12 from gp)
    lw t1, 0(t3)                        # load value for α
    mul t2, t0, t1                      # t@4 = α * α
    addi t3, gp, -24                    # addr of global t@4 (-24 from gp)
    sw t2, 0(t3)                        # store value to t@4
LQ15:
# Quad 15: -, 2, α, t@5
    li t0, 2                            # load literal value 2
    addi t3, gp, -12                    # addr of global α (-12 from gp)
    lw t1, 0(t3)                        # load value for α
    sub t2, t0, t1                      # t@5 = 2 - α
    addi t3, gp, -28                    # addr of global t@5 (-28 from gp)
    sw t2, 0(t3)                        # store value to t@5
LQ16:
# Quad 16: *, 2, α, t@6
    li t0, 2                            # load literal value 2
    addi t3, gp, -12                    # addr of global α (-12 from gp)
    lw t1, 0(t3)                        # load value for α
    mul t2, t0, t1                      # t@6 = 2 * α
    addi t3, gp, -32                    # addr of global t@6 (-32 from gp)
    sw t2, 0(t3)                        # store value to t@6
LQ17:
# Quad 17: -, t@5, t@6, t@7
    addi t3, gp, -28                    # addr of global t@5 (-28 from gp)
    lw t0, 0(t3)                        # load value for t@5
    addi t3, gp, -32                    # addr of global t@6 (-32 from gp)
    lw t1, 0(t3)                        # load value for t@6
    sub t2, t0, t1                      # t@7 = t@5 - t@6
    addi t3, gp, -36                    # addr of global t@7 (-36 from gp)
    sw t2, 0(t3)                        # store value to t@7
LQ18:
# Quad 18: /, t@4, t@7, t@8
    addi t3, gp, -24                    # addr of global t@4 (-24 from gp)
    lw t0, 0(t3)                        # load value for t@4
    addi t3, gp, -36                    # addr of global t@7 (-36 from gp)
    lw t1, 0(t3)                        # load value for t@7
    div t2, t0, t1                      # t@8 = t@4 / t@7
    addi t3, gp, -40                    # addr of global t@8 (-40 from gp)
    sw t2, 0(t3)                        # store value to t@8
LQ19:
# Quad 19: +, 2, t@8, t@9
    li t0, 2                            # load literal value 2
    addi t3, gp, -40                    # addr of global t@8 (-40 from gp)
    lw t1, 0(t3)                        # load value for t@8
    add t2, t0, t1                      # t@9 = 2 + t@8
    addi t3, gp, -44                    # addr of global t@9 (-44 from gp)
    sw t2, 0(t3)                        # store value to t@9
LQ20:
# Quad 20: :=, t@9, _, β
    addi t3, gp, -44                    # addr of global t@9 (-44 from gp)
    lw t0, 0(t3)                        # load value for t@9
    addi t3, gp, -16                    # addr of global β (-16 from gp)
    sw t0, 0(t3)                        # store value to β
LQ21:
# Quad 21: par, α, CV, _
    addi t3, gp, -12                    # addr of global α (-12 from gp)
    lw t0, 0(t3)                        # load value for α
    addi sp, sp, -4                     # make space on stack for CV param
    sw t0, 0(sp)                        # push CV value of 'α'
LQ22:
# Quad 22: par, β, REF, _
    addi t0, gp, -16                    # addr of global arg 'β' for REF
    addi sp, sp, -4                     # make space for REF param addr
    sw t0, 0(sp)                        # push REF addr for 'β'
LQ23:
# Quad 23: par, t@10, RET, _
    addi t0, gp, -48                    # addr of RET temp 't@10' in caller
    addi sp, sp, -4                     # make space for RET_VAL_ADDR
    sw t0, 0(sp)                        # push RET_VAL_ADDR
LQ24:
# Quad 24: call, αύξηση, _, _
    mv t0, gp                           # AL = Main's GP (main calls top-level)
    addi sp, sp, -4                     # make space for Access Link
    sw t0, 0(sp)                        # push Access Link
    jal L_ayxisi_entry                  # call αύξηση
    addi sp, sp, 16                     # pop params/ret_addr/AL (16 bytes)
LQ25:
# Quad 25: :=, t@10, _, γ
    addi t3, gp, -48                    # addr of global t@10 (-48 from gp)
    lw t0, 0(t3)                        # load value for t@10
    addi t3, gp, -20                    # addr of global γ (-20 from gp)
    sw t0, 0(t3)                        # store value to γ
LQ26:
# Quad 26: :=, 1, _, α
    li t0, 1                            # load literal value 1
    addi t3, gp, -12                    # addr of global α (-12 from gp)
    sw t0, 0(t3)                        # store value to α
LQ27:
# Quad 27: <=, α, 8, 29
    addi t3, gp, -12                    # addr of global α (-12 from gp)
    lw t0, 0(t3)                        # load value for α
    li t1, 8                            # load literal value 8
    ble t0, t1, LQ29                    # if α <= 8 goto LQ29
LQ28:
# Quad 28: jump, _, _, 33
    j LQ33                              # goto LQ33
LQ29:
# Quad 29: par, α, CV, _
    addi t3, gp, -12                    # addr of global α (-12 from gp)
    lw t0, 0(t3)                        # load value for α
    addi sp, sp, -4                     # make space on stack for CV param
    sw t0, 0(sp)                        # push CV value of 'α'
LQ30:
# Quad 30: call, τύπωσε_συν_1, _, _
    mv t0, gp                           # AL = Main's GP (main calls top-level)
    addi sp, sp, -4                     # make space for Access Link
    sw t0, 0(sp)                        # push Access Link
    jal L_typose_syn_1_entry            # call τύπωσε_συν_1
    addi sp, sp, 8                      # pop params/ret_addr/AL (8 bytes)
LQ31:
# Quad 31: +, α, 2, α
    addi t3, gp, -12                    # addr of global α (-12 from gp)
    lw t0, 0(t3)                        # load value for α
    li t1, 2                            # load literal value 2
    add t2, t0, t1                      # α = α + 2
    addi t3, gp, -12                    # addr of global α (-12 from gp)
    sw t2, 0(t3)                        # store value to α
LQ32:
# Quad 32: jump, _, _, 27
    j LQ27                              # goto LQ27
LQ33:
# Quad 33: :=, 1, _, β
    li t0, 1                            # load literal value 1
    addi t3, gp, -16                    # addr of global β (-16 from gp)
    sw t0, 0(t3)                        # store value to β
LQ34:
# Quad 34: <, β, 10, 36
    addi t3, gp, -16                    # addr of global β (-16 from gp)
    lw t0, 0(t3)                        # load value for β
    li t1, 10                           # load literal value 10
    blt t0, t1, LQ36                    # if β < 10 goto LQ36
LQ35:
# Quad 35: jump, _, _, 45
    j LQ45                              # goto LQ45
LQ36:
# Quad 36: <>, β, 22, 42
    addi t3, gp, -16                    # addr of global β (-16 from gp)
    lw t0, 0(t3)                        # load value for β
    li t1, 22                           # load literal value 22
    bne t0, t1, LQ42                    # if β <> 22 goto LQ42
LQ37:
# Quad 37: jump, _, _, 38
    j LQ38                              # goto LQ38
LQ38:
# Quad 38: >=, β, 23, 40
    addi t3, gp, -16                    # addr of global β (-16 from gp)
    lw t0, 0(t3)                        # load value for β
    li t1, 23                           # load literal value 23
    bge t0, t1, LQ40                    # if β >= 23 goto LQ40
LQ39:
# Quad 39: jump, _, _, 34
    j LQ34                              # goto LQ34
LQ40:
# Quad 40: <=, β, 24, 42
    addi t3, gp, -16                    # addr of global β (-16 from gp)
    lw t0, 0(t3)                        # load value for β
    li t1, 24                           # load literal value 24
    ble t0, t1, LQ42                    # if β <= 24 goto LQ42
LQ41:
# Quad 41: jump, _, _, 34
    j LQ34                              # goto LQ34
LQ42:
# Quad 42: +, β, 1, t@11
    addi t3, gp, -16                    # addr of global β (-16 from gp)
    lw t0, 0(t3)                        # load value for β
    li t1, 1                            # load literal value 1
    add t2, t0, t1                      # t@11 = β + 1
    addi t3, gp, -52                    # addr of global t@11 (-52 from gp)
    sw t2, 0(t3)                        # store value to t@11
LQ43:
# Quad 43: :=, t@11, _, β
    addi t3, gp, -52                    # addr of global t@11 (-52 from gp)
    lw t0, 0(t3)                        # load value for t@11
    addi t3, gp, -16                    # addr of global β (-16 from gp)
    sw t0, 0(t3)                        # store value to β
LQ44:
# Quad 44: jump, _, _, 34
    j LQ34                              # goto LQ34
LQ45:
# Quad 45: in, β, _, _
    li a7, 5                            # syscall: read_integer
    ecall
    addi t3, gp, -16                    # addr of global β (-16 from gp)
    sw a0, 0(t3)                        # store value to β
LQ46:
# Quad 46: +, β, 1, t@12
    addi t3, gp, -16                    # addr of global β (-16 from gp)
    lw t0, 0(t3)                        # load value for β
    li t1, 1                            # load literal value 1
    add t2, t0, t1                      # t@12 = β + 1
    addi t3, gp, -56                    # addr of global t@12 (-56 from gp)
    sw t2, 0(t3)                        # store value to t@12
LQ47:
# Quad 47: :=, t@12, _, β
    addi t3, gp, -56                    # addr of global t@12 (-56 from gp)
    lw t0, 0(t3)                        # load value for t@12
    addi t3, gp, -16                    # addr of global β (-16 from gp)
    sw t0, 0(t3)                        # store value to β
LQ48:
# Quad 48: <-, β, 100, 50
    addi t3, gp, -16                    # addr of global β (-16 from gp)
    lw t0, 0(t3)                        # load value for β
    li t1, 100                          # load literal value 100
    blt t0, t1, LQ50                    # if β <- 100 goto LQ50
LQ49:
# Quad 49: jump, _, _, 46
    j LQ46                              # goto LQ46
LQ50:
# Quad 50: halt, _, _, _
    li a0, 0                            # exit code 0
    li a7, 93                           # syscall: exit
    ecall
LQ51:
# Quad 51: end_block, τεστ, _, _
    addi sp, sp, 48                     # deallocate main's frame (48 bytes)