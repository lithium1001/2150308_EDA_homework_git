library(readxl)

# 读取 Excel 文件
file_path <- "D:/personal/2024/py/eda/2150308_EDA_homework/data/phone_parameters.xlsx"  # 确保文件路径正确
data <- read_excel(file_path)

# 数据描述
summary(data)

# 数据前几行
head(data)
