import json
import os
from datetime import datetime
from pathlib import Path


class TestNodeTool:


    def test_re_group(self):
        import re

        # 捕获分组示例
        pattern = r'(\d{2})-(\d{2})-(\d{4})'
        input_string = '31-12-2022'

        match = re.match(pattern, input_string)
        if match:
            day = match.group(1)
            month = match.group(2)
            year = match.group(3)
            print(f"Day: {day}, Month: {month}, Year: {year}")

        # 31-12-2022
        print(match.group(0))



    def test_regex_markdown(self):
        import re

        markdown_text = """---
id: note_20250118_120000_0
title: 项目进展
type: task_state
tags: [milestone, phase1]
created_at: 2025-01-18T12:00:00
updated_at: 2025-01-18T12:00:00
---

# 项目进展

已完成需求分析，下一步：设计方案

"""
        frontmatter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', markdown_text, re.DOTALL)

        frontmatter_text = frontmatter_match.group(1)
        content_start = frontmatter_match.end()

        """
id: note_20250118_120000_0
title: 项目进展
type: task_state
tags: [milestone, phase1]
created_at: 2025-01-18T12:00:00
updated_at: 2025-01-18T12:00:00
"""
        # print(frontmatter_text)

        """
        155 = 145 + 3（---） + 6（line break 换行）
        """
        # print(content_start)

        markdown_content = markdown_text[content_start:].strip()
        # '# 项目进展\n\n已完成需求分析，下一步：设计方案'
        print(markdown_content)

        # ['# 项目进展', '', '已完成需求分析，下一步：设计方案']
        # \n\n，所以这里会有一个''
        lines = markdown_content.split('\n')
        if lines and lines[0].startswith('# '):
            markdown_content = '\n'.join(lines[1:]).strip()

        print(markdown_content)


    def test_json_dump(self):
        workspace = os.path.join(os.path.dirname(__file__), 'note')
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)

        self.index_file = self.workspace / "notes_index.json"
        self.notes_index = {
            "notes": [],
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_notes": 0
            }
        }

        """notes_index.json
        {
          "notes": [],
          "metadata": {
            "created_at": "2026-02-06T10:54:09.275957",
            "total_notes": 0
          }
        }
        """
        # <_io.BufferedWriter name='C:\\VsCode\\llm\\hello-agents\\Co-creation-projects\\YYHDBL-HelloCodeAgentCli\\tools\\builtin\\note\\notes_index.json'>
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self.notes_index, f, ensure_ascii=False, indent=2)
