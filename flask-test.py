from flask import Flask
from openai import OpenAI
import os
import openpyxl
from flask import request


app = Flask(__name__)

base_url = "http://127.0.0.1:8000/v1/"
client = OpenAI(api_key="EMPTY", base_url=base_url)

@app.route('/test')
def index():
    securityCategory = request.args.get('securityCategory')
    controlPoint = request.args.get('controlPoint')
    quota = request.args.get('quota')
    resultRecord = request.args.get('resultRecord')
    # print(f"securityCategory值为{securityCategory}")
    # expectResult = getExpect("安全物理环境", "物理位置选择", "机房场地应选择在具有防震、防风和防雨等能力的建筑内")
    expectResult = getExpect(securityCategory, controlPoint, quota)
    chatResult = simple_chat(expectResult, resultRecord)
    result = resultDeal(chatResult)
    return result


def getExpect(securityCategory , controlPoint, quota):
    # 替换为您想要遍历的目录路径
    directory_path = './workingInstruction'
    
     # 遍历指定目录及其子目录下的所有文件
    for file in os.listdir(directory_path):
        # print(file)
        # 检查文件名是否以.xlsx结尾，并且是否包含目标字符串
        if file.endswith('.xlsx') and securityCategory in file:
            file_path = os.path.join(directory_path, file)

            # print(f"Reading file: {file_path}")
            break;

    # 打开工作簿
    workbook = openpyxl.load_workbook(file_path)
    sheet = workbook.active

    # 取消合并单元格
    unmerge_cells(sheet)

    # 初始化startIndex和endIndex
    startIndex, endIndex = None, None

    # 遍历第二列，寻找controlPoint
    for row in sheet.iter_rows(min_col=2, max_col=2, values_only=False):
        cell = row[0]  # 第二列的单元格
        if cell.value is not None and controlPoint in str(cell.value):  # 检查cell.value是否包含controlPoint
            if startIndex is None:
                startIndex = cell.row  # 记录开始索引
            endIndex = cell.row  # 每次找到时更新结束索引

    if startIndex is not None:
        print(f"开始索引：{startIndex}, 结束索引：{endIndex}")
    else:
        return "未找到控制点"


    # 如果找到了startIndex和endIndex，查找预期结果
    for row in sheet.iter_rows(min_col=1, max_col=5, min_row=startIndex, max_row=endIndex, values_only=False):
        cell2 = row[2]  # 第三列的单元格
        cell5 = row[4]  # 第五列的单元格
        # 检查配额和第五列的值
        if cell2.value is not None and quota in str(cell2.value) and cell5.value is not None:
            print(cell5.value)
            return cell5.value

def unmerge_cells(sheet):
    # 只对第二列的单元格进行操作，获取合并单元格范围的列表
    merged_cells = [cell for cell in sheet.merged_cells.ranges if cell.min_col == 2]

    # 遍历合并单元格并取消合并，同时填充内容
    for merged_cell in merged_cells:
        # 获取合并单元格的左上角单元格的值
        value = sheet.cell(row=merged_cell.min_row, column=merged_cell.min_col).value

        # 取消合并单元格
        sheet.unmerge_cells(str(merged_cell))

        # 填充取消合并后的所有单元格
        for row in range(merged_cell.min_row, merged_cell.max_row + 1):
            for col in range(merged_cell.min_col, merged_cell.max_col + 1):
                sheet.cell(row=row, column=col).value = value



def simple_chat(expectResult, result_record):
    use_stream=False
    messages = [
        {
            "role": "system",
            "content": f"""
            按照以下预期结果的几点规则，判断user输入的content是否符合:
            预期结果：
            {expectResult}
            """,
        },
        {
            "role": "user",
            "content": result_record
        }
    ]
    # print(messages)
    response = client.chat.completions.create(
        model="glm-4",
        messages=messages,
        stream=use_stream,
        max_tokens=256,
        temperature=0.4,
        presence_penalty=1.2,
        top_p=0.8,
    )
    if response:
        if use_stream:
            for chunk in response:
                print(chunk)
        else:
            return response
    else:
        print("Error:", response.status_code)


def resultDeal(chatResult):

    # 获取 choices 列表中的第一个 Choice 对象
    first_choice = chatResult.choices[0]

    # 获取 Choice 对象的 message 属性
    message = first_choice.message

    # 获取 message 的 content 字段内容
    content = message.content

    # 打印 content 字段的内容
    print(content)
    return content

if __name__ == "__main__":
    app.run(debug=True)
