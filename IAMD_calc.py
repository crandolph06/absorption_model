from random import random

def ShootShootLook(miss_1_pk, single_missile_type: bool, *miss_2_pk):
    rand_1 = random()
    rand_2 = random()
    miss_1_used = 0
    miss_2_used = 0
    # print(f'rand_1 is {rand_1} and rand_2 is {rand_2}')
    if rand_1 <= miss_1_pk:
        first_is_hit = True
    else:
        first_is_hit = False
    miss_1_used += 1
    if single_missile_type == False:
        if rand_2 <= miss_2_pk:
            second_is_hit = True
        else:
            second_is_hit = False
        miss_2_used += 1
    else:
        if rand_2 <= miss_1_pk:
            second_is_hit = True
        else:
            second_is_hit = False
        miss_1_used += 1
    if first_is_hit or second_is_hit == True:
        is_hit = True
    else:
        is_hit = False
    # print(f'first missile: {first_is_hit}, second missile: {second_is_hit}, total: {is_hit}')
    # print(f'miss_1_used: {miss_1_used}, miss_2_used: {miss_2_used}')
    return is_hit, (miss_1_used, miss_2_used)

def SingleShot(miss_pk):
    rand = random()
    if rand <= miss_pk:
        is_hit = True
    else:
        is_hit = False
    return is_hit

def RecursiveSingleShot(starting_int, current_int):
    missile_count = 0
    # missile is shot, intercept is successful (decrement int_inv, increment succ_eng) or unsuccessful (decrement int_inv, increment misses)
    # use new int_inv for new missile shot
    # stop when int_inv == 0 -- return missile_count, succ_eng, misses
    return missile_count

def VolleyCalc(int_1_start, pct_wrm_change, miss_1_pk, single_int_type: bool, *int_2_start, *miss_2_pk):
    missile_count = 0
    total_missed = 0
    total_succ_eng = 0
    current_int = int_1_start 
    if current_int >= (pct_wrm_change * int_1_start):
        if single_int_type == True:
            (is_hit, (miss_1_used, miss_2_used)) = ShootShootLook(miss_1_pk, True)
            if miss_2_used != 0:
                print(f'error: miss_2 should be 0')
                exit
            
            if is_hit == True:
                # good
            else:
                total_missed += 1
            current_int -= 
        else:
            is_hit = SingleShot(miss_1_pk)
    
    return missile_count

# is_hit = ShootShootLook(0.7, True)