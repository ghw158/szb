from adbutils import adb, AdbClient
import uiautomator2 as u2
import cv2
import numpy as np
import time
import ctypes
import sys
import os
import logging
import threading
import datetime
import json
from collections import defaultdict
import tkinter as tk
from tkinter import messagebox
import queue

# 配置日志 - 修复部分
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 创建文件日志处理器
file_handler = logging.FileHandler("script_log.log")
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# 创建控制台日志处理器
console_handler = logging.StreamHandler(sys.stdout)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# 添加处理器
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# 全局状态变量
script_running = True
script_paused = False
pause_logged = False  # 用于跟踪是否已记录暂停信息

# 回合统计相关变量
current_round_count = 1  # 当前对战的回合数
match_start_time = None  # 当前对战开始时间
match_history = []  # 存储所有对战记录
round_stats_file = "round_statistics.json"  # 统计数据保存文件
current_run_matches = 0  # 本次运行的对战次数
current_run_start_time = None  # 本次脚本启动时间

# 创建Tkinter消息队列
notification_queue = queue.Queue()

# 模板目录
TEMPLATES_DIR = "templates"

# 进化按钮模板（全局）
evolution_template = None
super_evolution_template = None

# 守护模板 （全局）
guard_template = None
super_guard_template = None

# 命令队列
command_queue = queue.Queue()

# 大爹
father_f_template = None
mother_f_template = None
father_h_template = None
mother_h_template = None
father_s_template = None
mother_s_template = None

# 法术
snake_god_rage = None

# 先后手
first_template = None
# second_template = None
if_first = True


def show_tkinter_notification(title, message):
    """使用Tkinter显示通知"""
    try:
        # 创建临时窗口
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口

        # 显示消息框
        messagebox.showinfo(title, message)

        # 关闭窗口
        root.destroy()
    except Exception as e:
        logger.error(f"显示Tkinter通知失败: {str(e)}")
        # 回退到传统弹窗
        ctypes.windll.user32.MessageBoxW(0, message, title, 0x40)


def notification_handler():
    """处理通知队列中的消息"""
    while True:
        try:
            # 从队列获取通知
            notification = notification_queue.get()
            if notification is None:  # 退出信号
                break

            title, message = notification
            show_tkinter_notification(title, message)
        except Exception as e:
            logger.error(f"通知处理出错: {str(e)}")
        finally:
            notification_queue.task_done()


def show_toast_notification(title, message, duration=5):
    """显示通知 - 使用Tkinter"""

    try:
        # 将通知放入队列
        notification_queue.put((title, message))
    except Exception as e:
        logger.error(f"添加通知到队列失败: {str(e)}")
        # 回退到传统弹窗
        ctypes.windll.user32.MessageBoxW(0, message, title, 0x40)


def connect_with_adbutils():
    client = AdbClient(host="127.0.0.1", port=5037)  # 默认ADB端口
    # *  这里填写你mumu的adb端口  右上角三条线→问题诊断→往下滑有个网络信息→adb调试端口
    client = adb.connect("127.0.0.1:16384")  # 常见模拟器端口
    devices = adb.device("127.0.0.1:16384")

    if not devices:
        raise RuntimeError("请检查 USB 调试是否开启")

    u2_device = u2.connect(devices.serial)
    return u2_device  # 返回 u2 设备对象


def take_screenshot():
    """获取设备截图"""
    try:
        # 获取设备对象
        device = adb.device("127.0.0.1:16384")
        if not device:
            logger.error("未找到可用设备")
            return None

        # 截图并返回
        return device.screenshot()

    except Exception as e:
        logger.error(f"截图失败: {str(e)}")
        return None


def load_template(templates_dir, filename):
    """加载模板图像并返回灰度图"""
    path = os.path.join(templates_dir, filename)
    if not os.path.exists(path):
        logger.error(f"模板文件不存在: {path}")
        return None

    template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        logger.error(f"无法加载模板: {path}")
    return template

def load_all_template():
    global guard_template
    if guard_template is None:
        guard_img = load_template(TEMPLATES_DIR, 'guard.png')
        guard_template = create_template_info(guard_img, "守护", threshold=0.28)

    global super_guard_template
    if super_guard_template is None:
        super_guard_img = load_template(TEMPLATES_DIR, 'super_guard.png')
        super_guard_template = create_template_info(super_guard_img, "超进化守护", threshold=0.5)

    global father_s_template
    if father_s_template is None:
        father_s_img = load_template(TEMPLATES_DIR, '543_s.png')
        father_s_template = create_template_info(father_s_img, "起始狼哥", threshold=0.8)

    global mother_s_template
    if mother_s_template is None:
        mother_s_img = load_template(TEMPLATES_DIR, '233_s.png')
        mother_s_template = create_template_info(mother_s_img, "起始233", threshold=0.8)

    global father_h_template
    if father_h_template is None:
        father_h_img = load_template(TEMPLATES_DIR, '543_h.png')
        father_h_template = create_template_info(father_h_img, "手牌狼哥", threshold=0.7)


    global mother_h_template
    if mother_h_template is None:
        mother_h_img = load_template(TEMPLATES_DIR, '233_h.png')
        mother_h_template = create_template_info(mother_h_img, "手牌233", threshold=0.7)

    global father_f_template
    if father_f_template is None:
        father_f_img = load_template(TEMPLATES_DIR, '543_f.png')
        father_f_template = create_template_info(father_f_img, "场上狼哥", threshold=0.7)

    global mother_f_template
    if mother_f_template is None:
        mother_f_img = load_template(TEMPLATES_DIR, '233_f.png')
        mother_f_template = create_template_info(mother_f_img, "场上233", threshold=0.7)

    global snake_god_rage
    if snake_god_rage is None:
        snake_god_rage_img = load_template(TEMPLATES_DIR, 'snake_god_rage.png')
        snake_god_rage = create_template_info(snake_god_rage_img, "蛇神之怒", threshold=0.6)

    global first_template
    if first_template is None:
        first_template_img = load_template(TEMPLATES_DIR, 'first.png')
        first_template = create_template_info(first_template_img, "先手", threshold=0.95)

    # global second_template
    # if second_template is None:
    #     second_template_img = load_template(TEMPLATES_DIR, 'second.png')
    #     second_template = create_template_info(second_template_img, "后手", threshold=0.85)



def create_template_info(template, name, threshold=0.85):
    """创建模板信息字典"""
    if template is None:
        return None

    h, w = template.shape
    return {
        'name': name,
        'template': template,
        'w': w,
        'h': h,
        'threshold': threshold
    }


def ask_user_question(question, default=True):
    """询问用户是/否问题"""
    print(f"{question} [{'Y/n' if default else 'y/N'}] ", end='')
    while True:
        response = input().strip().lower()

        if response == '':
            return default
        if response in ['y', 'yes']:
            return True
        if response in ['n', 'no']:
            return False

        print("请输入 y 或 n")


def command_listener():
    """监听控制台命令的线程函数（改进版）"""
    global script_running, script_paused, pause_logged

    logger.info("控制台命令监听已启动 (输入 'p'暂停, 'r'恢复, 'e'退出 或 's'统计)")

    # 添加命令提示
    print("\n>>> 命令提示: 'p'暂停, 'r'恢复, 'e'退出, 's'显示统计 <<<")
    print(">>> 输入命令后按回车 <<<\n")

    while script_running:
        try:
            # 直接使用input获取命令
            cmd = input("> ").strip().lower()

            # 将命令放入队列
            command_queue.put(cmd)

            # 添加处理反馈
            if cmd in ['p', 'r', 'e', 's']:
                print(f"命令 '{cmd}' 已接收")
            else:
                print(f"未知命令: '{cmd}'. 可用命令: p, r, e, s")

        except Exception as e:
            logger.error(f"命令监听出错: {str(e)}")
            time.sleep(1)  # 避免频繁出错

    logger.info("命令监听线程已退出")

def handle_command(cmd):
    """处理用户命令"""
    global script_running, script_paused, pause_logged

    if not cmd:
        return

    if cmd == "p":
        script_paused = True
        pause_logged = False
        logger.warning("用户请求暂停脚本")
        print(">>> 脚本已暂停 <<<")
    elif cmd == "r":
        script_paused = False
        pause_logged = False
        logger.info("用户请求恢复脚本")
        print(">>> 脚本已恢复 <<<")
    elif cmd == "e":
        script_running = False
        logger.info("正在退出脚本...")
        print(">>> 正在退出脚本... <<<")
    elif cmd == "s":
        show_round_statistics()
        print(">>> 已显示统计信息 <<<")
    else:
        logger.warning(f"未知命令: '{cmd}'. 可用命令:'p'暂停, 'r'恢复, 'e'退出 或 's'统计")
        print(f">>> 未知命令: '{cmd}' <<<")


def start_new_match():
    """开始新的对战"""
    global current_round_count, match_start_time, current_run_matches

    # 重置回合计数器
    current_round_count = 1
    match_start_time = time.time()
    current_run_matches += 1
    logger.info(f"===== 开始新的对战 =====")
    logger.info(f"本次运行对战次数: {current_run_matches}")

def detect_existing_match(gray_screenshot, templates):
    """检测脚本启动时是否已经处于对战状态"""
    global current_round_count, match_start_time, in_match

    # 检查是否已经在对战中（检测"结束回合"按钮或"敌方回合"提示）
    end_round_info = templates['end_round']
    enemy_round_info = templates['enemy_round']

    # 检测我方回合
    if end_round_info:
        max_loc, max_val = match_template(gray_screenshot, end_round_info)
        if max_val >= end_round_info['threshold']:
            in_match = True
            match_start_time = time.time()
            current_round_count = 1
            logger.info("脚本启动时检测到已处于我方回合，自动设置回合数为1")
            return True

    # 检测敌方回合
    if enemy_round_info:
        max_loc, max_val = match_template(gray_screenshot, enemy_round_info)
        if max_val >= enemy_round_info['threshold']:
            in_match = True
            match_start_time = time.time()
            current_round_count = 1
            logger.info("脚本启动时检测到已处于敌方回合，自动设置回合数为1")
            return True

    return False


def end_current_match():
    """结束当前对战并记录统计数据"""
    global current_round_count, match_start_time, match_history

    if match_start_time is None:
        return

    match_duration = time.time() - match_start_time
    minutes, seconds = divmod(match_duration, 60)

    match_record = {
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rounds": current_round_count,
        "duration": f"{int(minutes)}分{int(seconds)}秒",
        "run_id": current_run_start_time.strftime("%Y%m%d%H%M%S")
    }

    match_history.append(match_record)

    # 保存统计数据到文件
    save_round_statistics()

    logger.info(f"===== 对战结束 =====")
    logger.info(f"回合数: {current_round_count}, 持续时间: {int(minutes)}分{int(seconds)}秒")

    # 重置对战状态
    match_start_time = None
    current_round_count = 1


def save_round_statistics():
    """保存回合统计数据到文件"""
    try:
        with open(round_stats_file, 'w', encoding='utf-8') as f:
            json.dump(match_history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"保存统计数据失败: {str(e)}")


def load_round_statistics():
    """从文件加载回合统计数据"""
    global match_history

    if not os.path.exists(round_stats_file):
        return

    try:
        with open(round_stats_file, 'r', encoding='utf-8') as f:
            match_history = json.load(f)
    except Exception as e:
        logger.error(f"加载统计数据失败: {str(e)}")


def show_round_statistics():
    """显示回合统计数据"""
    if not match_history:
        logger.info("暂无对战统计数据")
        return

    # 计算总数据
    total_matches = len(match_history)
    total_rounds = sum(match['rounds'] for match in match_history)
    avg_rounds = total_rounds / total_matches if total_matches > 0 else 0

    # 计算本次运行数据
    current_run_matches = 0
    current_run_rounds = 0
    for match in match_history:
        if match.get('run_id') == current_run_start_time.strftime("%Y%m%d%H%M%S"):
            current_run_matches += 1
            current_run_rounds += match['rounds']

    current_run_avg = current_run_rounds / current_run_matches if current_run_matches > 0 else 0

    # 按回合数分组统计
    round_distribution = defaultdict(int)
    for match in match_history:
        round_distribution[match['rounds']] += 1

    # 显示统计数据
    logger.info(f"\n===== 对战回合统计 =====")
    logger.info(f"总对战次数: {total_matches}")
    logger.info(f"总回合数: {total_rounds}")
    logger.info(f"平均每局回合数: {avg_rounds:.1f}")

    # 显示本次运行统计
    logger.info(f"\n===== 本次运行统计 =====")
    logger.info(f"对战次数: {current_run_matches}")
    logger.info(f"总回合数: {current_run_rounds}")
    logger.info(f"平均每局回合数: {current_run_avg:.1f}")

    logger.info("\n回合数分布:")
    for rounds in sorted(round_distribution.keys()):
        count = round_distribution[rounds]
        percentage = (count / total_matches) * 100
        logger.info(f"{rounds}回合: {count}次 ({percentage:.1f}%)")

    # 显示最近5场对战
    logger.info("\n最近5场对战:")
    for match in match_history[-5:]:
        run_marker = "(本次运行)" if match.get('run_id') == current_run_start_time.strftime("%Y%m%d%H%M%S") else ""
        logger.info(f"{match['date']} - {match['rounds']}回合 ({match['duration']}) {run_marker}")


def curved_drag(device, start_x, start_y, end_x, end_y, duration=0.04, curve_factor=0.3):
    """
    执行弧线拖拽操作
    :param device: 设备对象
    :param start_x: 起始点x坐标
    :param start_y: 起始点y坐标
    :param end_x: 结束点x坐标
    :param end_y: 结束点y坐标
    :param duration: 拖拽持续时间（秒）
    :param curve_factor: 曲线因子，控制曲线的弯曲程度（0-1）
    """
    # 计算中点
    mid_x = (start_x + end_x) / 2
    mid_y = (start_y + end_y) / 2

    # 固定方向
    direction = 1

    # 计算两点之间的距离
    distance = ((end_x - start_x) ** 2 + (end_y - start_y) ** 2) ** 0.5

    # 计算曲线高度（基于距离和曲线因子）
    curve_height = distance * curve_factor * direction

    # 计算垂直方向向量
    dx = end_x - start_x
    dy = end_y - start_y

    # 计算垂直方向（旋转90度）
    if dx == 0 and dy == 0:
        # 如果起点和终点相同，使用默认方向
        perpendicular_x, perpendicular_y = 0, 1
    else:
        # 计算垂直向量
        magnitude = (dx * dx + dy * dy) ** 0.5
        normalized_dx = dx / magnitude
        normalized_dy = dy / magnitude
        perpendicular_x = -normalized_dy
        perpendicular_y = normalized_dx

    # 计算控制点坐标
    control_x = mid_x + perpendicular_x * curve_height
    control_y = mid_y + perpendicular_y * curve_height

    # 计算曲线上的点（贝塞尔曲线）
    steps = 20  # 曲线上的点数

    # 执行拖拽操作
    device.swipe(start_x, start_y, end_x, end_y, duration)

def load_evolution_template():
    """加载进化按钮模板"""
    global evolution_template
    if evolution_template is None:
        # 使用全局定义的 TEMPLATES_DIR
        template_img = load_template(TEMPLATES_DIR, 'evolution.png')
        if template_img is None:
            logger.error("无法加载进化按钮模板")
            return None

        evolution_template = create_template_info(
            template_img,
            "进化按钮",
            threshold=0.85
        )
    return evolution_template


def load_super_evolution_template():
    """加载超进化按钮模板"""
    global super_evolution_template
    if super_evolution_template is None:
        # 使用全局定义的 TEMPLATES_DIR
        template_img = load_template(TEMPLATES_DIR, 'super_evolution.png')
        if template_img is None:
            logger.error("无法加载超进化按钮模板")
            return None

        super_evolution_template = create_template_info(
            template_img,
            "超进化按钮",
            threshold=0.85
        )
    return super_evolution_template


def match_template(gray_image, template_info):
    """执行模板匹配并返回结果"""
    if not template_info:
        return None, 0

    result = cv2.matchTemplate(gray_image, template_info['template'], cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    return max_loc, max_val

def detect_evolution_button(gray_screenshot):
    """检测进化按钮是否出现"""
    evolution_info = load_evolution_template()
    if not evolution_info:
        return None, 0

    max_loc, max_val = match_template(gray_screenshot, evolution_info)
    return max_loc, max_val


def detect_super_evolution_button(gray_screenshot):
    """检测超进化按钮是否出现"""
    evolution_info = load_super_evolution_template()
    if not evolution_info:
        return None, 0

    max_loc, max_val = match_template(gray_screenshot, evolution_info)
    return max_loc, max_val


def start_hands(device):
    global father_s_template, mother_s_template
    global if_first
    xlist = []
    screenshot = take_screenshot()
    screenshot_np = np.array(screenshot)
    screenshot_cv = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
    gray_screenshot = cv2.cvtColor(screenshot_cv, cv2.COLOR_BGR2GRAY)

    # 裁剪指定区域
    x1, y1, x2, y2 = 1060, 560, 1250, 675
    region = gray_screenshot[y1:y2, x1:x2]

    # 模板匹配
    res = cv2.matchTemplate(region, first_template['template'], cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    logger.info(max_val)

    if max_val >= first_template['threshold']:
        if_first = True
        logger.info("你是先手")
    else:
        if_first = False
        logger.info("你是后手")

    all_boxes = []
    all_scores = []

    if father_s_template:
        
        result = cv2.matchTemplate(gray_screenshot, father_s_template['template'], cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= father_s_template['threshold'])
        for pt in zip(*loc[::-1]):
            x, y = pt
            w, h = father_s_template['w'], father_s_template['h']
            all_boxes.append([x, y, x + w, y + h])
            all_scores.append(result[y, x])

    if mother_s_template:
        result = cv2.matchTemplate(gray_screenshot, mother_s_template['template'], cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= mother_s_template['threshold'])
        for pt in zip(*loc[::-1]):
            x, y = pt
            w, h = mother_s_template['w'], mother_s_template['h']
            all_boxes.append([x, y, x + w, y + h])
            all_scores.append(result[y, x])

    if all_boxes:
        boxes_np = np.array(all_boxes)
        scores_np = np.array(all_scores)
        keep_indices = nms(boxes_np, scores_np, threshold=0.5)
        for i in keep_indices:
            x1, y1, x2, y2 = all_boxes[i]
            x_pos = int((x1 + x2) // 2)
            xlist.append(x_pos)

    pos = []

    for x in xlist:
        if x >= 169 and x < 379:
            pos.append(1)
        if x >= 379 and x < 571:
            pos.append(2)
        if x >= 571 and x < 780:
            pos.append(3)
        if x >= 780 and x < 990:
            pos.append(4)

    if 1 not in pos:
        curved_drag(device, 274, 531, 274, 200, 0.04)
    if 2 not in pos:
        curved_drag(device, 480, 531, 480, 200, 0.04)
    if 3 not in pos:
        curved_drag(device, 685, 531, 685, 200, 0.04)
    if 4 not in pos:
        curved_drag(device, 880, 531, 880, 200, 0.04)


def if_father(device):
    global father_h_template, mother_h_template
    screenshot = take_screenshot()
    screenshot_np = np.array(screenshot)
    screenshot_cv = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
    gray_screenshot = cv2.cvtColor(screenshot_cv, cv2.COLOR_BGR2GRAY)

    max_loc1, max_val1 = match_template(gray_screenshot, father_h_template)
    if max_val1 >= father_h_template['threshold']:
        center_x = max_loc1[0] + father_h_template['w'] // 2
        center_y = max_loc1[1] + father_h_template['h'] // 2
        curved_drag(device, center_x, center_y, center_x, 200, 0.04)
        time.sleep(0.1)

    max_loc2, max_val2 = match_template(gray_screenshot, mother_h_template)
    if max_val2 >= mother_h_template['threshold']:
        center_x = max_loc2[0] + mother_h_template['w'] // 2
        center_y = max_loc2[1] + mother_h_template['h'] // 2
        curved_drag(device, center_x, center_y, center_x, 200, 0.04)
        time.sleep(0.1)
    time.sleep(0.1)
    # device.click(976, 675)


def evolution_father(follower_positions):
    global father_f_template, mother_f_template
    pos = []
    screenshot = take_screenshot()
    screenshot_np = np.array(screenshot)
    screenshot_cv = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
    gray_screenshot = cv2.cvtColor(screenshot_cv, cv2.COLOR_BGR2GRAY)

    max_loc1, max_val1 = match_template(gray_screenshot, father_f_template)
    if max_val1 >= father_f_template['threshold']:
        center_x = max_loc1[0] + father_f_template['w'] // 2
        center_y = max_loc1[1] + father_f_template['h'] // 2
        if center_y >= 320:
            pos.append((center_x, center_y))

    max_loc2, max_val2 = match_template(gray_screenshot, mother_f_template)
    if max_val2 >= mother_f_template['threshold']:
        center_x = max_loc2[0] + mother_f_template['w'] // 2
        center_y = max_loc2[1] + mother_f_template['h'] // 2
        if center_y >= 320:
            pos.append((center_x, center_y))
    follower_positions[:0] = pos
    return follower_positions


def detect_spell(list):
    global snake_god_rage
    xlist = []
    screenshot = take_screenshot()
    screenshot_np = np.array(screenshot)
    screenshot_cv = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
    gray_screenshot = cv2.cvtColor(screenshot_cv, cv2.COLOR_BGR2GRAY)

    all_boxes = []
    all_scores = []

    if snake_god_rage:
        result = cv2.matchTemplate(gray_screenshot, snake_god_rage['template'], cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= snake_god_rage['threshold'])
        for pt in zip(*loc[::-1]):
            x, y = pt
            w, h = snake_god_rage['w'], snake_god_rage['h']
            all_boxes.append([x, y, x + w, y + h])
            all_scores.append(result[y, x])

    if all_boxes:
        boxes_np = np.array(all_boxes)
        scores_np = np.array(all_scores)
        keep_indices = nms(boxes_np, scores_np, threshold=0.5)
        for i in keep_indices:
            x1, y1, x2, y2 = all_boxes[i]
            x_pos = int((x1 + x2) // 2)
            xlist.append(x_pos)

    for del_center in xlist:
        list = [x for x in list if abs(x - del_center) >= 50]
    if len(list) != 0:
        logger.info("检测到法术")

    return list

def use_spell(device):
    global snake_god_rage
    screenshot = take_screenshot()
    screenshot_np = np.array(screenshot)
    screenshot_cv = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
    gray_screenshot = cv2.cvtColor(screenshot_cv, cv2.COLOR_BGR2GRAY)

    max_loc, max_val = match_template(gray_screenshot, snake_god_rage)
    if max_val >= snake_god_rage['threshold']:
        center_x = max_loc[0] + snake_god_rage['w'] // 2
        center_y = max_loc[1] + snake_god_rage['h'] // 2
        curved_drag(device, center_x, center_y, center_x, 200, 0.04)
        time.sleep(0.5)
        device.click(645, 63)
        time.sleep(1)
        device.click(976, 679)
        time.sleep(0.1)


def nms(boxes, scores, threshold=0.5):
    """
    非极大值抑制(Non-Maximum Suppression)
    :param boxes: 检测框列表，格式为[[x1, y1, x2, y2], ...]
    :param scores: 每个检测框的得分
    :param threshold: 重叠阈值
    :return: 保留的检测框索引
    """
    if len(boxes) == 0:
        return []

    # 获取每个框的坐标
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # 计算每个框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # 按得分排序(从高到低)
    indices = np.argsort(scores)

    keep = []
    while len(indices) > 0:
        # 取出当前得分最高的框
        last = len(indices) - 1
        i = indices[last]
        keep.append(i)

        # 计算当前框与其他框的重叠区域
        xx1 = np.maximum(x1[i], x1[indices[:last]])
        yy1 = np.maximum(y1[i], y1[indices[:last]])
        xx2 = np.minimum(x2[i], x2[indices[:last]])
        yy2 = np.minimum(y2[i], y2[indices[:last]])

        # 计算重叠区域的宽高和面积
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / areas[indices[:last]]

        # 删除重叠度过高的框
        indices = np.delete(indices, np.concatenate(([last], np.where(overlap > threshold)[0])))

    return keep


def detect_guard(frames_to_sample=3, frame_interval=0.1):
    """多帧投票机制检测敌方守护随从"""
    global guard_template, super_guard_template
    opp_xlist_all = []

    for _ in range(frames_to_sample):
        screenshot = take_screenshot()
        if not screenshot:
            continue

        screenshot_np = np.array(screenshot)
        screenshot_cv = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
        gray_screenshot = cv2.cvtColor(screenshot_cv, cv2.COLOR_BGR2GRAY)

        all_boxes = []
        all_scores = []

        # 普通守护
        if guard_template:
            result = cv2.matchTemplate(gray_screenshot, guard_template['template'], cv2.TM_CCOEFF_NORMED)
            loc = np.where(result >= guard_template['threshold'])
            for pt in zip(*loc[::-1]):
                x, y = pt
                w, h = guard_template['w'], guard_template['h']
                all_boxes.append([x, y, x + w, y + h])
                all_scores.append(result[y, x])

        # 超进化守护
        if super_guard_template:
            result = cv2.matchTemplate(gray_screenshot, super_guard_template['template'], cv2.TM_CCOEFF_NORMED)
            loc = np.where(result >= super_guard_template['threshold'])
            for pt in zip(*loc[::-1]):
                x, y = pt
                w, h = super_guard_template['w'], super_guard_template['h']
                all_boxes.append([x, y, x + w, y + h])
                all_scores.append(result[y, x])

        # NMS + 过滤敌方区域
        if all_boxes:
            boxes_np = np.array(all_boxes)
            scores_np = np.array(all_scores)
            keep_indices = nms(boxes_np, scores_np, threshold=0.5)
            for i in keep_indices:
                x1, y1, x2, y2 = all_boxes[i]
                x_pos = int((x1 + x2) // 2)
                if y1 <= 240:
                    opp_xlist_all.append(x_pos)

        # 间隔一小段时间再检测下一帧
        time.sleep(frame_interval)

    # 简单投票机制：x坐标相近的视为同一个目标
    clustered_x = []
    for x in opp_xlist_all:
        found_cluster = False
        for group in clustered_x:
            if abs(x - group[0]) < 25:
                group.append(x)
                found_cluster = True
                break
        if not found_cluster:
            clustered_x.append([x])

    final_xlist = [int(sum(g)/len(g)) for g in clustered_x if len(g) >= 2]  # 至少出现2次才算有效
    final_xlist.sort()

    return final_xlist

def detect_creatures(debug=False):
    """基于差分与对称检测双方随从位置（包含特殊随从），返回中心x坐标列表"""
    you_xlist, opp_xlist = [], []

    screenshot = take_screenshot()
    if screenshot is None:
        print("无法截取屏幕")
        return you_xlist, opp_xlist

    screenshot_np = np.array(screenshot)
    screenshot_cv = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
    field_roi = screenshot_cv[136:480, 200:1070]  # 场地区域裁剪

    field_bg = cv2.imread("./templates/field.png", cv2.IMREAD_GRAYSCALE)
    if field_bg is None:
        print("无法读取背景图 field.png")
        return you_xlist, opp_xlist

    field_roi_gray = cv2.cvtColor(field_roi, cv2.COLOR_BGR2GRAY)
    field_bg_resized = cv2.resize(field_bg, (field_roi_gray.shape[1], field_roi_gray.shape[0]))
    diff = cv2.absdiff(field_roi_gray, field_bg_resized)
    _, diff_bin = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)

    # 膨胀操作增强结构
    kernel_size = 5
    cv2.dilate(diff_bin, (kernel_size, kernel_size), diff_bin, iterations=8)

    if debug:
        cv2.imwrite("debug_diff.png", diff)
        cv2.imwrite("debug_diff_bin.png", diff_bin)
        color_debug = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
        cv2.line(color_debug, (0, 163 - 136), (diff_bin.shape[1], 163 - 136), (0, 0, 255), 1)
        cv2.line(color_debug, (0, 333 - 136), (diff_bin.shape[1], 333 - 136), (255, 0, 0), 1)
        cv2.line(color_debug, (0, 153 - 136), (diff_bin.shape[1], 153 - 136), (0, 255, 255), 1)
        cv2.line(color_debug, (0, 323 - 136), (diff_bin.shape[1], 323 - 136), (255, 255, 0), 1)
        cv2.imwrite("debug_diff_lines.png", color_debug)

    def find_creatures(scan_y, step, width_limit):
        x = 0
        coords = []
        width = diff_bin.shape[1]
        while x < width:
            if diff_bin[scan_y, x] > 0:
                x1 = x
                x_probe = min(x1 + step, width - 1)
                while x_probe > x1 and diff_bin[scan_y, x_probe] == 0:
                    x_probe -= 1
                center_x = 200 + (x1 + x_probe) // 2
                coords.append(center_x)
                x = x_probe + width_limit
            else:
                x += 1
        return coords

    # 检测各类随从
    opp_normal = find_creatures(163 - 136, step=120, width_limit=10)
    opp_special = find_creatures(153 - 136, step=150, width_limit=10)
    you_normal = find_creatures(333 - 136, step=120, width_limit=10)
    you_special = find_creatures(323 - 136, step=150, width_limit=10)

    def merge_and_fix(xlist):
        xlist = [x for x in xlist if 275 <= x <= 1024]
        xlist.sort()
        result = []
        i = 0
        while i < len(xlist):
            curr = xlist[i]
            j = i + 1
            while j < len(xlist) and abs(xlist[j] - curr) < 100:
                curr = (curr + xlist[j]) // 2
                j += 1
            result.append(curr)
            i = j
        # 补漏
        i = 0
        while i < len(result) - 1:
            if result[i + 1] - result[i] > 270:
                missed = (result[i + 1] + result[i]) // 2
                result.insert(i + 1, missed)
            i += 1
        return sorted(result)

    opp_xlist = merge_and_fix(opp_normal + opp_special)
    you_xlist = merge_and_fix(you_normal + you_special)

    return you_xlist, opp_xlist


def perform_evolution_actions(device, is_super=False):
    """
    执行进化/超进化操作（带检测）
    :param device: 设备对象
    :param is_super: 是否为超进化操作
    """
    evolution_detected = False  # 添加此变量

    # 进化可能的位置（按优先级排序）
    follower_positions = [
        (644, 402), (482, 402), (551, 402),
        (315, 402), (731, 402), (811, 402), (974, 402)
    ]

    # 优先检查关键随从
    follower_positions = evolution_father(follower_positions)

    # 遍历每个随从位置
    for pos in follower_positions:

        # 使用固定位置
        follower_x, follower_y = pos
        device.click(follower_x, follower_y)
        time.sleep(0.1)

        # 获取截图检测进化按钮
        screenshot = take_screenshot()
        if screenshot is None:
            continue

        # 转换为OpenCV格式
        screenshot_np = np.array(screenshot)
        screenshot_cv = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
        gray_screenshot = cv2.cvtColor(screenshot_cv, cv2.COLOR_BGR2GRAY)

        # 根据进化类型选择检测函数
        if is_super:
            max_loc, max_val = detect_super_evolution_button(gray_screenshot)
            button_type = "超进化"
        else:
            max_loc, max_val = detect_evolution_button(gray_screenshot)
            button_type = "进化"

        if max_val >= 0.80:  # 检测阈值
            # 计算按钮中心位置
            if is_super:
                template_info = load_super_evolution_template()
            else:
                template_info = load_evolution_template()

            if template_info:
                center_x = max_loc[0] + template_info['w'] // 2
                center_y = max_loc[1] + template_info['h'] // 2
                device.click(center_x, center_y)
                logger.info(f"检测到{button_type}按钮并点击")
                evolution_detected = True
                time.sleep(0.3)

    return evolution_detected


def perform_full_actions(device):
    """720P分辨率下的出牌攻击操作（使用弧线拖拽）"""
    # 投币
    device.click(1167, 537)
    time.sleep(0.1)
    # 展牌
    device.click(976, 675)
    time.sleep(0.2)

    if_father(device)


    # 5次出牌拖拽（使用弧线）
    start_y = 672
    end_y = 400
    duration = 0.1
    drag_points_x = [405, 551, 684, 830, 959]

    # drag_points_x = detect_spell(drag_points_x)

    for x in drag_points_x:
        curved_drag(device, x, start_y, x, end_y, duration)
        time.sleep(0.1)  # 固定等待时间

    time.sleep(0.2)


    atk_flag = True
    atk_num = 0
    loop_count = 0

    # 部分攻击坐标（使用弧线）
    start_y = 402
    end_x = 646
    end_y = 64

    you_xlist, opp_xlist = detect_creatures()
    logger.info(f"检测结果: 我方随从x坐标={you_xlist}, 敌方随从x坐标={opp_xlist}")
    max_atk_count = len(you_xlist)
    g_xlist = detect_guard()
    logger.info(f"检测结果: 敌方守护随从x坐标={g_xlist}")
    if len(g_xlist) == 0:
        for start_x in you_xlist:
            curved_drag(device, start_x, start_y, end_x, end_y, 0.1)
            time.sleep(0.1)
    else:
        while atk_flag:
            you_xlist, opp_xlist = detect_creatures()
            logger.info(f"检测结果: 我方随从x坐标={you_xlist}, 敌方随从x坐标={opp_xlist}")
            time.sleep(0.1)
            g_xlist = detect_guard()
            logger.info(f"检测结果: 敌方守护随从x坐标={g_xlist}")
            if len(you_xlist) != 0:
                if len(g_xlist) != 0:
                    curved_drag(device, you_xlist[(-1 - atk_num) % len(you_xlist)], start_y, g_xlist[0], 238, 0.1)
                    loop_count += 1
                    time.sleep(1)
                    you_xlist1, opp_xlist1 = detect_creatures()
                    if len(you_xlist1) == len(you_xlist):
                        logger.info(f"我方随从数量没有变化")
                        atk_num += 1
                else:
                    curved_drag(device, you_xlist[(-1 - atk_num) % len(you_xlist)], start_y, end_x, end_y, 0.1)
                    loop_count += 1
                    atk_num += 1
            else:
                atk_flag = False
            if loop_count == max_atk_count:
                atk_flag = False
            time.sleep(0.5)
    time.sleep(0.1)


def perform_fullS_actions(device):
    """720P分辨率下执行进化与攻击操作"""
    # 投币
    device.click(1167, 537)
    time.sleep(0.1)
    # 展牌
    device.click(976, 675)
    time.sleep(0.2)

    if_father(device)

    # 5次出牌拖拽（使用弧线）
    start_y = 672
    end_y = 400
    duration = 0.1
    drag_points_x = [405, 551, 684, 830, 959]

    # drag_points_x = detect_spell(drag_points_x)

    for x in drag_points_x:
        curved_drag(device, x, start_y, x, end_y, duration)
        time.sleep(0.1)

    time.sleep(0.2)

    # 执行进化操作（带检测）
    evolution_performed = perform_evolution_actions(device, is_super=False)

    if not evolution_performed:
        logger.warning("未检测到进化按钮，尝试备用方案")
        # 备用方案：尝试点击固定位置
        evolution_button = (160, 288)
        device.click(*evolution_button)
        evolution_button = (313, 285)
        device.click(*evolution_button)
        time.sleep(0.4)

    # 等待最终进化动画完成
    time.sleep(3)

    atk_flag = True
    atk_num = 0
    loop_count = 0

    # 部分攻击坐标（使用弧线）
    start_y = 402
    end_x = 646
    end_y = 64

    you_xlist, opp_xlist = detect_creatures()
    logger.info(f"检测结果: 我方随从x坐标={you_xlist}, 敌方随从x坐标={opp_xlist}")
    max_atk_count = len(you_xlist)
    g_xlist = detect_guard()
    logger.info(f"检测结果: 敌方守护随从x坐标={g_xlist}")
    if len(g_xlist) == 0:
        for start_x in you_xlist:
            curved_drag(device, start_x, start_y, end_x, end_y, 0.1)
            time.sleep(0.1)
    else:
        while atk_flag:
            you_xlist, opp_xlist = detect_creatures()
            logger.info(f"检测结果: 我方随从x坐标={you_xlist}, 敌方随从x坐标={opp_xlist}")
            time.sleep(0.1)
            g_xlist = detect_guard()
            logger.info(f"检测结果: 敌方守护随从x坐标={g_xlist}")
            if len(you_xlist) != 0:
                if len(g_xlist) != 0:
                    curved_drag(device, you_xlist[(-1 - atk_num) % len(you_xlist)], start_y, g_xlist[0], 238, 0.1)
                    loop_count += 1
                    time.sleep(1)
                    you_xlist1, opp_xlist1 = detect_creatures()
                    if len(you_xlist1) == len(you_xlist):
                        logger.info(f"我方随从数量没有变化")
                        atk_num += 1
                else:
                    curved_drag(device, you_xlist[(-1 - atk_num) % len(you_xlist)], start_y, end_x, end_y, 0.1)
                    loop_count += 1
                    atk_num += 1
            else:
                atk_flag = False
            if loop_count == max_atk_count:
                atk_flag = False
            time.sleep(0.5)
    time.sleep(0.1)


def perform_fullPlus_actions(device):
    """720P分辨率下执行超进化与攻击操作"""
    # 投币
    device.click(1167, 537)
    time.sleep(0.1)
    # 展牌
    device.click(976, 675)
    time.sleep(0.2)

    # use_spell(device)
    # use_spell(device)
    # use_spell(device)

    if_father(device)

    # 5次出牌拖拽（使用弧线）
    start_y = 672
    end_y = 400
    duration = 0.1



    drag_points_x = [405, 551, 684, 830, 959]
    for x in drag_points_x:
        curved_drag(device, x, start_y, x, end_y, duration)
        time.sleep(0.1)

    time.sleep(0.2)

    # 执行超进化操作（带检测）
    evolution_performed = perform_evolution_actions(device, is_super=True)

    if not evolution_performed:
        logger.warning("未检测到超进化按钮，尝试备用方案")
        # 备用方案：尝试点击固定位置
        evolution_button = (313, 285)
        device.click(*evolution_button)
        time.sleep(0.1)
        

    # 等待最终进化动画完成
    time.sleep(3)

    atk_flag = True
    atk_num = 0
    loop_count = 0

    # 部分攻击坐标（使用弧线）
    start_y = 402
    end_x = 646
    end_y = 64

    you_xlist, opp_xlist = detect_creatures()
    logger.info(f"检测结果: 我方随从x坐标={you_xlist}, 敌方随从x坐标={opp_xlist}")
    max_atk_count = len(you_xlist)
    g_xlist = detect_guard()
    logger.info(f"检测结果: 敌方守护随从x坐标={g_xlist}")
    if len(g_xlist) == 0:
        for start_x in you_xlist:
            curved_drag(device, start_x, start_y, end_x, end_y, 0.1)
            time.sleep(0.1)
    else:
        while atk_flag:
            you_xlist, opp_xlist = detect_creatures()
            logger.info(f"检测结果: 我方随从x坐标={you_xlist}, 敌方随从x坐标={opp_xlist}")
            time.sleep(0.1)
            g_xlist = detect_guard()
            logger.info(f"检测结果: 敌方守护随从x坐标={g_xlist}")
            if len(you_xlist) != 0:
                if len(g_xlist) != 0:
                    curved_drag(device, you_xlist[(-1 - atk_num) % len(you_xlist)], start_y, g_xlist[0], 238, 0.1)
                    loop_count += 1
                    time.sleep(1)
                    you_xlist1, opp_xlist1 = detect_creatures()
                    if len(you_xlist1) == len(you_xlist):
                        logger.info(f"我方随从数量没有变化")
                        atk_num += 1
                else:
                    curved_drag(device, you_xlist[(-1 - atk_num) % len(you_xlist)], start_y, end_x, end_y, 0.1)
                    loop_count += 1
                    atk_num += 1
            else:
                atk_flag = False
            if loop_count == max_atk_count:
                atk_flag = False
            time.sleep(0.5)
    time.sleep(0.1)

def main():
    global script_running, script_paused, pause_logged
    global current_round_count, match_start_time, current_run_matches, current_run_start_time
    global in_match, evolution_template, super_evolution_template
    global if_first
    has_swapped_hands = False

    # 初始化进化模板
    evolution_template = None
    super_evolution_template = None

    # 启动通知处理线程
    notification_thread = threading.Thread(target=notification_handler)
    notification_thread.daemon = False
    notification_thread.start()

    # 记录脚本启动时间
    current_run_start_time = datetime.datetime.now()
    current_run_matches = 0
    in_match = False  # 是否在对战中

    # 配置参数
    SCAN_INTERVAL = 2  # 主循环间隔(秒)

    logger.info("===== 脚本开始运行 =====")
    logger.info(f"脚本启动时间: {current_run_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 加载历史统计数据
    load_round_statistics()

    # 1. 加载所有模板
    logger.info("正在加载模板...")

    templates = {
        'decision': create_template_info(load_template(TEMPLATES_DIR, 'decision.png'), "决定"),
        'end_round': create_template_info(load_template(TEMPLATES_DIR, 'end_round.png'), "结束回合"),
        'enemy_round': create_template_info(load_template(TEMPLATES_DIR, 'enemy_round.png'), "敌方回合"),
        'end': create_template_info(load_template(TEMPLATES_DIR, 'end.png'), "结束"),
        'war': create_template_info(load_template(TEMPLATES_DIR, 'war.png'), "决斗"),
        'retry': create_template_info(load_template(TEMPLATES_DIR, 'error_retry.png'), "重试"),
        'rank': create_template_info(load_template(TEMPLATES_DIR, 'rank.png'), "升段"),
        'mainPage': create_template_info(load_template(TEMPLATES_DIR, 'mainPage.png'), "游戏主页面"),
        'MuMuPage': create_template_info(load_template(TEMPLATES_DIR, 'MuMuPage.png'), "MuMu主页面"),
        'LoginPage': create_template_info(load_template(TEMPLATES_DIR, 'LoginPage.png'), "排队主界面"),
        'enterGame': create_template_info(load_template(TEMPLATES_DIR, 'enterGame.png'), "排队进入"),
        'missionCompleted': create_template_info(load_template(TEMPLATES_DIR, 'missionCompleted.png'), "任务完成"),
        'notContinue': create_template_info(load_template(TEMPLATES_DIR, 'No.png'), "不继续中断的对战"),
        'Ok': create_template_info(load_template(TEMPLATES_DIR, 'Ok.png'), "好的"),
        'rankUp': create_template_info(load_template(TEMPLATES_DIR, 'rankUp.png'), "阶位提升"),
        'groupUp': create_template_info(load_template(TEMPLATES_DIR, 'groupUp.png'), "分组升级"),
        'backTitle': create_template_info(load_template(TEMPLATES_DIR, 'backTitle.png'), "分组升级"),
    }

    load_all_template()

    logger.info("模板加载完成")


    # 2. 连接设备
    logger.info("正在连接设备...")
    try:
        device = connect_with_adbutils()
        logger.info("设备连接成功")
    except Exception as e:
        logger.error(f"设备连接失败: {str(e)}")
        return

    # 3. 启动命令监听线程
    cmd_thread = threading.Thread(target=command_listener, daemon=True)
    cmd_thread.start()

    # 4. 检测脚本启动时是否已经在对战中
    logger.info("检测当前游戏状态...")
    init_screenshot = take_screenshot()
    if init_screenshot is not None:
        # 转换为OpenCV格式
        init_screenshot_np = np.array(init_screenshot)
        init_screenshot_cv = cv2.cvtColor(init_screenshot_np, cv2.COLOR_RGB2BGR)
        gray_init_screenshot = cv2.cvtColor(init_screenshot_cv, cv2.COLOR_BGR2GRAY)

        # 检测是否已经在游戏中
        if detect_existing_match(gray_init_screenshot, templates):
            # 设置本次运行的对战次数
            current_run_matches = 1
            logger.info(f"本次运行对战次数: {current_run_matches} (包含已开始的对战)")
        else:
            logger.info("未检测到进行中的对战")
    else:
        logger.warning("无法获取初始截图，跳过状态检测")

    # 5. 主循环
    logger.info("脚本初始化完成，开始运行...")

    try:
        while script_running:
            start_time = time.time()

            # 检查命令队列
            while not command_queue.empty():
                cmd = command_queue.get()
                handle_command(cmd)

            # 检查脚本暂停状态
            if script_paused:
                # 如果尚未记录暂停信息，则记录一次
                if not pause_logged:
                    logger.info("脚本暂停中...输入 'r' 继续")
                    pause_logged = True

                # 在暂停状态下每1秒检查一次
                time.sleep(1)
                continue

            # 获取截图
            screenshot = take_screenshot()
            if screenshot is None:
                time.sleep(SCAN_INTERVAL)
                continue

            # 转换为OpenCV格式
            screenshot_np = np.array(screenshot)
            screenshot_cv = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
            gray_screenshot = cv2.cvtColor(screenshot_cv, cv2.COLOR_BGR2GRAY)

            # 检查其他按钮
            button_detected = False
            for key in ['war', 'end', 'decision', 'end_round', 'retry', 'rank', 'mainPage', 'MuMuPage', 'LoginPage',
                        'enterGame', 'missionCompleted', 'notContinue', 'Ok','groupUp', 'rankUp', 'backTitle']:  # 修改检测顺序，优先处理war
                template_info = templates[key]
                if not template_info:
                    continue

                max_loc, max_val = match_template(gray_screenshot, template_info)


                if max_val >= template_info['threshold']:
                    # 记录当前回合信息（在点击前）
                    if key == 'end_round' and in_match:
                        logger.info(f"已发现'结束回合'按钮 (当前回合: {current_round_count})")

                    if key == 'decision' and in_match and not has_swapped_hands:
                        # 检测到“决定”按钮，执行换牌
                        start_hands(device)
                        has_swapped_hands = True  # 设置换牌已完成

                    # 处理对战开始/结束逻辑
                    if key == 'war':
                        # 检测到"决斗"按钮，表示新对战开始
                        has_swapped_hands = False   # 重置换牌标志
                        if_first = True     # 重置先后手标志
                        if in_match:
                            # 如果已经在战斗中，先结束当前对战
                            end_current_match()
                            logger.info("检测到新对战开始，结束上一场对战")

                        # 开始新的对战
                        start_new_match()
                        in_match = True
                        logger.info("检测到新对战开始")

                    elif key == 'end_round' and in_match and if_first:
                        # 检测到"结束回合"按钮，增加回合计数
                        if current_round_count in (5, 6):  # 第5，6回合
                            logger.info(f"第{current_round_count}回合，执行进化")
                            perform_fullS_actions(device)
                        elif current_round_count in (7, 8, 9):  # 第7,8,9回合
                            logger.info(f"第{current_round_count}回合，执行超进化")
                            perform_fullPlus_actions(device)
                        else:  # 其他回合
                            logger.info(f"第{current_round_count}回合，执行正常操作")
                            perform_full_actions(device)
                        current_round_count += 1

                    elif key == 'end_round' and in_match and not if_first:
                        # 检测到"结束回合"按钮，增加回合计数
                        if current_round_count in (4, 5):  # 第4 ，5回合
                            logger.info(f"第{current_round_count}回合，执行进化")
                            perform_fullS_actions(device)
                        elif current_round_count in (6, 7, 8):  # 第6,7,8回合
                            logger.info(f"第{current_round_count}回合，执行超进化")
                            perform_fullPlus_actions(device)
                        else:  # 其他回合
                            logger.info(f"第{current_round_count}回合，执行正常操作")
                            perform_full_actions(device)
                        current_round_count += 1


                    # 计算中心点并点击
                    center_x = max_loc[0] + template_info['w'] // 2
                    center_y = max_loc[1] + template_info['h'] // 2
                    device.click(center_x, center_y)

                    button_detected = True
                    # 点击后短暂延迟
                    time.sleep(0.5)  # 固定等待时间
                    break  # 点击一个按钮后跳出循环

            # 计算处理时间并调整等待
            process_time = time.time() - start_time
            sleep_time = max(0, SCAN_INTERVAL - process_time)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("用户中断脚本执行")
    except Exception as e:
        logger.exception("脚本运行异常:")
    finally:
        # 结束当前对战（如果正在进行）
        if in_match:
            end_current_match()

        # 设置运行标志为False
        script_running = False

        # 保存统计数据
        save_round_statistics()

        # 关闭命令线程
        if 'cmd_thread' in locals() and cmd_thread.is_alive():
            try:
                # 对于Windows系统，发送一个虚拟按键来中断kbhit()
                if os.name == 'nt':
                    import msvcrt
                    # 发送一个回车键来中断等待
                    ctypes.windll.user32.keybd_event(0x0D, 0, 0, 0)
                    ctypes.windll.user32.keybd_event(0x0D, 0, 0x0002, 0)
                # 关闭标准输入以中断 input() 调用
                sys.stdin.close()
                logger.debug("已关闭标准输入")
            except Exception as e:
                logger.error(f"关闭标准输入时出错: {str(e)}")

            # 等待线程退出
            cmd_thread.join(timeout=2.0)
            if cmd_thread.is_alive():
                logger.warning("命令线程未能正常退出")

        # 关闭通知线程
        notification_queue.put(None)
        if 'notification_thread' in locals() and notification_thread.is_alive():
            notification_thread.join(timeout=2.0)
            if notification_thread.is_alive():
                logger.warning("通知线程未能正常退出")

        # 显示最终统计数据
        show_round_statistics()

        # 计算本次运行时间
        run_duration = datetime.datetime.now() - current_run_start_time
        hours, remainder = divmod(run_duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)

        # 结束信息
        logger.info("\n===== 本次运行总结 =====")
        logger.info(f"脚本启动时间: {current_run_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"运行时长: {int(hours)}小时{int(minutes)}分钟{int(seconds)}秒")
        logger.info(f"完成对战次数: {current_run_matches}")
        logger.info("===== 脚本结束运行 =====")


if __name__ == "__main__":
    main()
