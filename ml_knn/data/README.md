# 用户数据目录

## 数据格式说明

### CSV 格式

将数据保存为 `input.csv`，格式如下：

```csv
feature1,feature2,feature3,target
1.2,3.4,5.6,0
2.3,4.5,6.7,1
3.4,5.6,7.8,2
...
```

- 特征列：数值型特征
- 目标列：默认名称为 `target` 或 `label`，若不同需通过 `--target_col` 参数指定

### Excel 格式

将数据保存为 `input.xlsx`，默认读取第一个 sheet，格式与 CSV 相同。

## 使用示例

```bash
# 使用 CSV 数据
python main.py --data_path data/input.csv --target_col label

# 使用 Excel 数据
python main.py --data_path data/input.xlsx --target_col target
```

## 注意事项

1. 特征必须是数值型（连续或离散）
2. 目标列必须是分类标签（整数或字符串）
3. 缺失值将自动填充（均值填充）
4. 特征将自动标准化
