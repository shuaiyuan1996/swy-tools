import numpy as np
import scipy.optimize as opt

## configs
ZHEN_AND_YU_FOOD_ONLY = True


## default buff
# 烹饪时间 = int(原时间 * (1 - 熟练度加成 - 食魂加成 - 口碑加成 - 画院加成))
# 烹饪售价 = 原收益 * (1 + 熟练度加成 + 迎宾加成) 
buff_time_shihun = 0.3
buff_time_koubei = 0.01
buff_time_huayuan = 0.013
buff_time_general = buff_time_shihun + buff_time_koubei + buff_time_huayuan

buff_price_yingbin = 0.03
buff_price_general = buff_price_yingbin

max_cook_num = 33
kitchen_space_num = 5


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def read_menu(file_path):
    with open(file_path, 'r') as fin:
        title = fin.readline()[:-1].split(',')
        items = []
        for line in fin.readlines():
            items.append(line[:-1].split(','))

        item_name = [item[0] for item in items]
        item_level = np.array([item[1] for item in items], dtype=int)
        item_ori_price = np.array([item[2] for item in items], dtype=float)
        item_ori_time = np.array([item[3] for item in items], dtype=float)
        item_price_buff = np.array([item[4] for item in items], dtype=float)
        item_time_buff = np.array([item[5] for item in items], dtype=float)
        item_ingred = np.array([item[6:] for item in items], dtype=float)    

    # compute actual cooking price and time
    item_price = (item_ori_price * (1 + buff_price_general + item_price_buff)).astype(float)
    item_time = (item_ori_time * (1 - buff_time_general - item_time_buff)).astype(float)

    return item_name, item_level, item_ingred, item_price, item_time


def filter_item_level(item_name, item_level, item_ingred, item_price, item_time):
    valid = item_level >= 2
    item_name = np.array(item_name)[valid].tolist()
    item_level = item_level[valid]
    item_ingred = item_ingred[valid]
    item_price = item_price[valid]
    item_time = item_time[valid]
    return item_name, item_level, item_ingred, item_price, item_time


def solver(item_name, item_level, item_ingred, item_price, item_time, ingred_cnt, total_time):
    """
    We solve this problem by integer linear programming.

    Notations:
    p_i: price of item i
    t_i: cooking time of item i
    a_ij: the number of ingredient j needed for each item i
    m_j: the total number of ingredient j available
    T: total time allowed
    x_i: variable, the number of item i
    z_i: variable, the number of slots assigned to item i

    Problem formulation:
    min     f({x_i}, {z_i}) = - sum_i (p_i * x_i)
    s. t.   sum_i (x_i * a_ij) <= m_j, for all j    # constraints on resources
            x_i * t_i <= T * z_i, for all i         # constraints on cooking time (*may have an issue if x_i cannot be evenly divided by z_i*)
            x_i <= 33 * z_i, for all i              # each slot can take at most 33 items
            sum_i z_i <= 5                          # 5 cooking slots in total
            0 <= x_i, for all i                     # nonnegative number of items
            all {x_i}, {z_i} are integers

    """
    print("当前算法版本：scipy.optimize.milp for integer linear programming")

    item_cnt = len(item_name)

    # objective function
    c = np.hstack((- item_price, np.zeros(item_cnt)))

    # constraints
    A1 = np.hstack((item_ingred.T, np.zeros((6, item_cnt))))
    b1_l = np.full(6, - np.inf)
    b1_u = ingred_cnt

    A2 = np.hstack((- np.diag(item_time), np.diag(np.full(item_cnt, total_time))))
    b2_l = np.zeros(item_cnt)
    b2_u = np.full(item_cnt, np.inf)

    A3 = np.hstack((- np.diag(np.ones(item_cnt)), np.diag(np.full(item_cnt, max_cook_num))))
    b3_l = np.zeros(item_cnt)
    b3_u = np.full(item_cnt, np.inf)

    A4 = np.hstack((np.zeros(item_cnt), np.ones(item_cnt)))[None]
    b4_l = - np.inf
    b4_u = kitchen_space_num

    A = np.vstack((A1, A2, A3, A4))
    b_l = np.hstack((b1_l, b2_l, b3_l, b4_l))
    b_u = np.hstack((b1_u, b2_u, b3_u, b4_u))

    constraints = opt.LinearConstraint(A, b_l, b_u)

    var_bound_l = np.zeros(item_cnt * 2)
    var_bound_u = np.hstack((np.full(item_cnt, kitchen_space_num * max_cook_num), np.full(item_cnt, kitchen_space_num)))
    bounds = opt.Bounds(lb=var_bound_l, ub=var_bound_u)

    integrality = np.ones(item_cnt * 2)

    res = opt.milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality, options={'time_limit': 300})

    #import IPython; IPython.embed(); exit()
    x_star = res.x[:item_cnt]
    nonzero = x_star >= 1

    plan_item_name = np.array(item_name)[nonzero].tolist()
    plan_item_cnt = x_star[nonzero].astype(int)
    plan_item_time = item_time[nonzero]
    plan_item_price = item_price[nonzero]
    plan_item_ingred = item_ingred[nonzero]

    return plan_item_name, plan_item_cnt, plan_item_time, plan_item_price, plan_item_ingred


def print_plan(plan_item_name, plan_item_cnt, plan_item_time, plan_item_price, plan_item_ingred):
    plan_time = plan_item_cnt * plan_item_time
    plan_price = plan_item_cnt * plan_item_price
    order = np.argsort(plan_price)[::-1].astype(int)

    print("----------------")
    print(bcolors.OKCYAN + "规划结果" + bcolors.ENDC + ":")
    for i, idx in enumerate(order.tolist()):
        time = int(np.ceil(plan_time[idx]))
        price = int(plan_price[idx])
        line = "  {0:2d}. ".format(i+1)
        line += bcolors.WARNING + "{0:s}".format(plan_item_name[idx]) + bcolors.ENDC + " x {0:d}: ".format(int(plan_item_cnt[idx]))
        line += "总价值 " + bcolors.WARNING + "{0:d}".format(price) + bcolors.ENDC + " 贝币, "
        line += "总耗时 " + bcolors.WARNING + "{0:02d}:{1:02d}:{2:02d}".format(time // 3600, (time % 3600) // 60, time % 60) + bcolors.ENDC + "。"
        print(line)

    time = int(np.ceil(np.max(plan_time)))
    line = "以上总价值 " + bcolors.WARNING + "{0:d}".format(int(np.sum(plan_price))) + bcolors.ENDC + " 贝币, "
    line += "总耗时 " + bcolors.WARNING + "{0:02d}:{1:02d}:{2:02d}".format(time // 3600, (time % 3600) // 60, time % 60) + bcolors.ENDC + "。"
    print(line)

    ingred_cost = plan_item_ingred.T @ plan_item_cnt
    print(bcolors.OKCYAN + "\n剩余原材料(剩余数量/总数量)" + bcolors.ENDC + ":")
    line = []
    for i in range(len(ingred_name)):
        line.append(ingred_name[i] + "({0:d}/{1:d})".format(int(ingred_cnt[i] - ingred_cost[i]), int(ingred_cnt[i])))
    print(', '.join(line))

    return


if __name__ == '__main__':

    print(bcolors.ENDC + "================")
    print(bcolors.HEADER + "感谢使用此程序！我们将为您计算最优的做菜规划。" + bcolors.ENDC + "作者：Kolin")
    print("本次计算采用以下默认加成。如有需要，可以自行修改源代码。")
    print(bcolors.OKCYAN)
    print("烹饪时间加成：{0:3.1f}~{1:3.1f}%".format((buff_time_general + 0.1) * 100, (buff_time_general + 0.2) * 100))
    print("    1. 满熟练度加成：10~20%")
    print("    2. 烹饪食魂加成：{0:2.0f}%".format(buff_time_shihun * 100))
    print("    3. 口碑加成：{0:2.0f}%".format(buff_time_koubei * 100))
    print("    4. 画院加成：{0:3.1f}%".format(buff_time_huayuan * 100))
    print("烹饪价格加成：{0:3.1f}%".format((buff_price_general + 0.1) * 100))
    print("    1. 满熟练度加成：10%")
    print("    2. 迎宾食魂加成：{0:2.0f}%".format(buff_price_yingbin * 100))
    print("单次烹饪最高数量：{0:2d}".format(max_cook_num))
    print(bcolors.ENDC)

    ## read data table
    item_name, item_level, item_ingred, item_price, item_time = read_menu('../data/kitchen_menu.txt')
    ingred_name = ["蔬果", "肉类", "谷物", "蛋类", "河鲜", "海鲜"]
    print("菜单数据载入完成!")

    ## input current resources and desired time
    ingred_cnt = input("请按顺序（" + '、'.join(ingred_name) + "）输入当前食材数量，以空格隔开：" + bcolors.WARNING + bcolors.UNDERLINE)
    ingred_cnt = np.array(ingred_cnt.split(' '), dtype=int)
    #ingred_cnt = np.array([7638, 6426, 4505, 4509, 1812, 935], dtype=float)
    #ingred_cnt = np.array([3260, 32600, 16300, 0, 0, 1630], dtype=float)

    total_time_input = input(bcolors.ENDC + "请输入限制烹饪时长（单位：小时），若无限制请回车跳过：" + bcolors.WARNING + bcolors.UNDERLINE)
    if total_time_input == '':
    	total_time = 1e8
    else:
    	total_time = float(total_time_input) * 3600.

    if total_time < 0:
        total_time = 1e8

    ## Preprocessing: simplify the table
    if ZHEN_AND_YU_FOOD_ONLY:
        print(bcolors.ENDC + "(为简便计算，我们将只考虑珍菜和御菜)")
        item_name, item_level, item_ingred, item_price, item_time = \
            filter_item_level(item_name, item_level, item_ingred, item_price, item_time)


    ## Start to solve!
    print(bcolors.ENDC + "规划中，请稍候...")

    # test the lower bound
    plan_item_name, plan_item_cnt, plan_item_time, plan_item_price, plan_item_ingred = \
        solver(item_name, item_level, item_ingred, item_price, item_time, ingred_cnt, total_time)

    # split for items that need more than one slot
    final_item_name = []
    final_item_cnt = np.zeros(kitchen_space_num)
    final_item_time = np.zeros(kitchen_space_num)
    final_item_price = np.zeros(kitchen_space_num)
    final_item_ingred = np.zeros((kitchen_space_num, 6))
    cnt = 0

    for i in range(len(plan_item_name)):
        num_slot = int(np.ceil(plan_item_cnt[i] / max_cook_num))
        for j in range(num_slot):
            final_item_name.append(plan_item_name[i])
            final_item_cnt[cnt + j] = np.floor(plan_item_cnt[i] / num_slot)
            final_item_time[cnt + j] = plan_item_time[i]
            final_item_price[cnt + j] = plan_item_price[i]
            final_item_ingred[cnt + j] = plan_item_ingred[i]

        for j in range(int(plan_item_cnt[i]) % num_slot):
            final_item_cnt[cnt + j] += 1

        cnt += num_slot

    print_plan(final_item_name, final_item_cnt, final_item_time, final_item_price, final_item_ingred)

    print(bcolors.OKGREEN + "Success!\n" + bcolors.ENDC)
    # import IPython; IPython.embed()


