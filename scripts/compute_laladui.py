import numpy as np

with open('../data/level_up_price.txt', 'r') as fin:
    prices = fin.readlines()
    prices = np.array(prices, dtype=int)

star_lvl = [20, 40, 60, 80]
star_prices = [20000, 50000, 100000, 200000]
star_nuts = [20, 60, 160, 300]

## read inputs
cur_list = input("请输入当前食魂等级：")
cur_list = np.array(cur_list.split(' '), dtype=int)
num_char = len(cur_list)

group_size = int(input("请输入拉拉队人数："))

target = int(input("请输入目标总等级："))


## solve it!
char_idx = cur_list.argsort()[-group_size:]
char_cur_lvl = cur_list[char_idx]

# find the characters that need to level up
char_lvl_up = np.zeros(group_size, dtype=bool)
for i in range(group_size):
    if target > char_cur_lvl[i] * i + char_cur_lvl[i:].sum():
        char_lvl_up[i] = True

# write the levels for those DO NOT need to level up
num_char_lvl_up = char_lvl_up.sum()
char_target_lvl = np.zeros(group_size)
char_target_lvl[num_char_lvl_up:] = char_cur_lvl[num_char_lvl_up:]

# write the (integer) levels for those DO need to level up
lvl_left = max(target - char_cur_lvl[num_char_lvl_up:].sum(), 0)
avg_lvl_left = lvl_left // num_char_lvl_up
res_lvl_left = lvl_left % num_char_lvl_up
char_target_lvl[:num_char_lvl_up] = avg_lvl_left
char_target_lvl[(num_char_lvl_up - res_lvl_left):num_char_lvl_up] += 1

char_cur_lvl = char_cur_lvl.astype(int)
char_target_lvl = char_target_lvl.astype(int)

# compute the shell coins needed
cost = 0
nut_cost = np.zeros(len(star_lvl))
for i in range(num_char_lvl_up):
    cost += prices[(char_cur_lvl[i] - 1):(char_target_lvl[i]-1)].sum()

    for j in range(len(star_lvl)):
        st, st_p = star_lvl[j], star_prices[j]
        if char_cur_lvl[i] < st and st < char_target_lvl[i]:
            cost += st_p
            nut_cost[j] += 1

## output
print("\n------- 计算结果 -------")
print("当前食魂最高总等级: {}".format(char_cur_lvl.sum()))

selection = []
for i in range(num_char):
    if i in char_idx:
        group_idx = np.where(char_idx == i)[0][0]
        if char_cur_lvl[group_idx] == char_target_lvl[group_idx]:
            selection.append("{}(√)".format(char_cur_lvl[group_idx]))
        else:
            selection.append("{}(->{})".format(char_cur_lvl[group_idx], char_target_lvl[group_idx]))
    else:
        selection.append("{}(x)".format(cur_list[i]))
print("请选择食魂：" + "  ".join(selection))
print(" (x 代表不选择；√ 代表选择且无需提升；-> 代表选择且需提升等级)\n")

if num_char_lvl_up == 0:
    print("您的等级已经足够！")
else:
    resources = ["{} 贝币".format(cost)]
    
    if nut_cost[0] > 0:
        resources.append("{} 良坚果".format(int(nut_cost[0] * star_nuts[0])))
    if nut_cost[1] > 0:
        resources.append("{} 尚坚果".format(int(nut_cost[1] * star_nuts[1])))
    if nut_cost[2] > 0:
        resources.append("{} 珍坚果".format(int(nut_cost[2] * star_nuts[2])))
    if nut_cost[3] > 0:
        resources.append("{} 御坚果".format(int(nut_cost[3] * star_nuts[3])))

    print("提升所需资源：" + "、".join(resources))

for s in star_lvl:
    if (char_cur_lvl == s).sum() > 0:
        print("* 注：当前已经20/40/60/80级的食魂，我们默认您已经升星。若未升星，则还需加上这一部分升星的成本哦！")
        break

print("------------------------\n")

