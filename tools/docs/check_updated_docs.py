#!/usr/bin/env python3

import os
import re
import sys
from pathlib import Path
from collections import defaultdict


def find_changed_source_files(changed_files_str):
    if not changed_files_str:
        return []
    
    changed = changed_files_str.replace('\n', ' ').split()
    return [f for f in changed if f.startswith('src/')]


def extract_code_paths_from_docs(docs_dir='docs'):
    code_to_docs = defaultdict(set)
    
    for md_file in Path(docs_dir).rglob('*.md'):
        try:
            content = md_file.read_text(encoding='utf-8')
        except Exception:
            continue
        
        paths = set()
        for match in re.finditer(r'`([a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]+)`', content):
            path = match.group(1)
            if path.startswith('src/'):
                paths.add(path)
        
        for match in re.finditer(r'\|\s*`([a-zA-Z0-9_\-./]+)`\s*\|', content):
            path = match.group(1)
            if path.startswith('src/'):
                paths.add(path)
        
        for match in re.finditer(r'\[([^\]]+)\]\((\.\./)?([a-zA-Z0-9_\-./]+\.[a-zA-Z0-9]+)\)', content):
            path = match.group(3)
            if path.startswith('src/'):
                paths.add(path)
        
        for path in paths:
            code_to_docs[path].add(str(md_file.relative_to(Path(docs_dir).parent)))
    
    return code_to_docs


def find_related_docs(changed_files, code_to_docs):
    related = defaultdict(set)
    
    for changed_file in changed_files:
        changed_parts = changed_file.split('/')
        
        for doc_code_path, docs in code_to_docs.items():
            doc_parts = doc_code_path.split('/')
            
            if doc_parts == changed_parts[:len(doc_parts)]:
                for doc in docs:
                    related[doc].add(changed_file)
            elif doc_code_path == changed_file:
                for doc in docs:
                    related[doc].add(changed_file)
    
    return dict(related)


def generate_pr_comment(related_docs):
    if not related_docs:
        return None
    
    lines = [
        '## 📚 Documentation Update',
        '',
        'The following documentation may need to be updated based on code changes:',
        ''
    ]
    
    for doc_file in sorted(related_docs.keys()):
        changed_files = related_docs[doc_file]
        lines.append(f'- **[{doc_file}]({doc_file})**')
        for code_file in sorted(changed_files):
            lines.append(f'  - Related code: `{code_file}`')
        lines.append('')
    
    lines.append('*Please review these files for necessary documentation updates*')
    
    return '\n'.join(lines)


if __name__ == '__main__':
    changed_files_input = sys.argv[1] if len(sys.argv) > 1 else ''
    
    changed_files = find_changed_source_files(changed_files_input)
    
    if not changed_files:
        sys.exit(0)
    
    code_to_docs = extract_code_paths_from_docs()
    related_docs = find_related_docs(changed_files, code_to_docs)
    
    comment = generate_pr_comment(related_docs)
    
    if comment:
        print(comment)
        sys.exit(0)
    
    sys.exit(1)
