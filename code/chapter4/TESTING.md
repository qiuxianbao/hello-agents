# 测试文档

### 运行所有测试
```bash
python -m pytest test_*.py -v
```

### 运行单个测试文件
```bash
python -m pytest test_dotenv.py -v
```

### 运行带覆盖率的测试
```bash
#pip install pytest-cov
python test_dotenv.py coverage
```

```bash
# 在CI环境中运行
#pip install pytest-html
#tb=traceback 用于指定回溯信息的展示级别。short=简洁模式
python -m pytest --tb=short

# 生成JUnit XML报告
#python -m pytest --junitxml=test-results.xml

# 生成JUnit HTML报告
python -m pytest --html=test-results.html
```