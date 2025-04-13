import numpy as np
import json
import os
import re
import time
from openai import OpenAI

# 创建输出目录
os.makedirs('output', exist_ok=True)

# 从环境变量中获取API密钥
api_key = os.environ.get("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("请设置DEEPSEEK_API_KEY环境变量")

# 初始化OpenAI客户端
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

# 从数据集中解析并提取数独谜题和解答
def parse_sudoku_data(file_path, num_samples=200, min_clues=78):
    """
    解析包含跨行数据的数独数据集，只选择线索数大于等于min_clues的样本
    """
    data = []
    with open(file_path, 'r') as f:
        # 跳过表头
        header = f.readline().strip()
        
        # 读取所有的行并处理
        content = f.read()
        
        # 使用正则表达式匹配每一行的模式
        # 模式：quizzes(81个数字),solutions(81个数字),clue_numbers(数字)
        pattern = r'([0-9]{81}),([0-9]{81}),([0-9]+)'
        matches = re.findall(pattern, content)
        
        # 筛选出符合线索数要求的样本
        filtered_matches = []
        for match in matches:
            if len(match) == 3:
                quizzes, solutions, clue_numbers = match
                if int(clue_numbers) >= min_clues:
                    filtered_matches.append(match)
        
        print(f"从{len(matches)}个样本中筛选出{len(filtered_matches)}个线索数大于等于{min_clues}的样本")
        
        # 随机采样，确保有足够的样本
        import random
        if len(filtered_matches) > num_samples:
            filtered_matches = random.sample(filtered_matches, num_samples)
        
        for match in filtered_matches:
            quizzes, solutions, clue_numbers = match
            if len(quizzes) == 81 and len(solutions) == 81:
                data.append({
                    'quizzes': quizzes,
                    'solutions': solutions,
                    'clue_numbers': int(clue_numbers)
                })
    
    print(f"成功解析了{len(data)}个数独数据，线索数均大于等于{min_clues}")
    return data

# 将数独字符串转换为指定格式
def format_sudoku(puzzle_str):
    """将81个字符的数独字符串转换为9x9矩阵格式"""
    # 确保输入字符串长度为81
    if len(puzzle_str) != 81:
        print(f"警告: 数独字符串长度不是81，而是{len(puzzle_str)}，无法格式化。")
        return None

    # 将字符串转换为9x9矩阵
    grid = np.array(list(puzzle_str)).reshape(9, 9)
    
    # 格式化输出
    formatted = []
    for i in range(9):
        row = " ".join(grid[i, :3]) + " | " + " ".join(grid[i, 3:6]) + " | " + " ".join(grid[i, 6:])
        formatted.append(row)
        if i == 2 or i == 5:
            formatted.append("------+-------+------")
    
    return "\n".join(formatted)

# 调用DeepSeek API
def call_deepseek_api(formatted_puzzle):
    """调用DeepSeek API解决数独并获取推理过程"""
    prompt = f"""以下是一个数独游戏，在9乘9的81宫格中，数字的顺序分别为：
{formatted_puzzle}
其中0代表空缺的数字，需要你去填写，请你完成这个数独游戏，并输出相同格式的答案，答案中不需要其他字符！"""
    
    messages = [{"role": "user", "content": prompt}]
    
    try:
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=messages
        )
        
        reasoning_content = response.choices[0].message.reasoning_content
        content = response.choices[0].message.content
        
        return reasoning_content, content
    except Exception as e:
        print(f"API调用出错: {e}")
        return None, None

# 创建数独推理数据集
def create_dataset(input_file, output_file, num_samples=200, min_clues=78):
    """创建数独推理数据集"""
    # 解析数据集
    data = parse_sudoku_data(input_file, num_samples, min_clues)
    
    # 创建结果列表
    results = []
    
    # 处理每个数独谜题
    for idx, item in enumerate(data):
        print(f"处理第 {idx+1}/{len(data)} 个数独，线索数: {item['clue_numbers']}...")
        
        puzzle = item['quizzes']
        solution = item['solutions']
        clue_number = item['clue_numbers']
        
        # 格式化数独谜题
        formatted_puzzle = format_sudoku(puzzle)
        if not formatted_puzzle:
            print(f"跳过索引 {idx}，无法格式化谜题")
            continue
        
        # 调用DeepSeek API
        reasoning, answer = call_deepseek_api(formatted_puzzle)
        
        if reasoning and answer:
            # 创建数据条目
            entry = {
                "question": f"""以下是一个数独游戏，在9乘9的81宫格中，数字的顺序分别为：
{formatted_puzzle}
其中0代表空缺的数字，需要你去填写，请你完成这个数独游戏，并输出相同格式的答案。""",
                "answer": f"<think>{reasoning}</think>\n\n<answer>{answer}</answer>",
                "original_puzzle": puzzle,
                "original_solution": solution,
                "clue_number": clue_number
            }
            
            results.append(entry)
            
            # 每处理10个样本，保存一次中间结果
            if len(results) % 10 == 0:
                with open(f'output/intermediate_{len(results)}.json', 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"已保存中间结果，当前处理了 {len(results)} 个样本")
            
            # API调用间隔，避免频率限制
            time.sleep(2)
        else:
            print(f"API调用失败，跳过索引 {idx}")
        
    # 保存最终数据集
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"数据集创建完成，共{len(results)}个样本，已保存至 {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='创建数独推理数据集')
    parser.add_argument('--samples', type=int, default=200, help='要处理的样本数量')
    parser.add_argument('--input', type=str, default='dataset/sudoku_cluewise.csv', help='输入数据集路径')
    parser.add_argument('--output', type=str, default='output/sudoku_reasoning_dataset.json', help='输出数据集路径')
    parser.add_argument('--min_clues', type=int, default=78, help='最小线索数量')
    
    args = parser.parse_args()
    
    print(f"开始创建数据集，将处理 {args.samples} 个样本，线索数大于等于 {args.min_clues}")
    create_dataset(args.input, args.output, num_samples=args.samples, min_clues=args.min_clues) 