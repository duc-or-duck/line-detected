with open('src/pipeline.py', 'r', encoding='utf-8') as f:
    content = f.read()

old = '        \"\"Convert normalized x(y) linear fit into image-space y = slope*x + intercept.\"\"\"'
new = '        """Convert normalized x(y) linear fit into image-space y = slope*x + intercept."""'
if old in content:
    content = content.replace(old, new, 1)
    print('Fixed docstring')
else:
    print('String not found, checking raw...')
    for i, line in enumerate(content.split('\n')):
        if 'Convert normalized' in line:
            print(f'Line {i+1}: {repr(line)}')

with open('src/pipeline.py', 'w', encoding='utf-8') as f:
    f.write(content)

