import unicodedata

p = 'env_check.py'
t = open(p, encoding='utf-8').read()
lines = t.splitlines()
curly = '\u2018\u2019\u201c\u201d\u201a\u201b\u02bc\u00b4\uff07\u300c\u300d'
hits = []
for i, line in enumerate(lines, 1):
    for j, ch in enumerate(line):
        if ch in curly or (ord(ch) > 127 and unicodedata.category(ch) in ('Pi', 'Pf')):
            hits.append((i, j, ch, unicodedata.name(ch, '?')))
print('可疑引号:', hits[:20])
for ln in (112, 113, 114, 115):
    line = lines[ln - 1]
    print(ln, '单引号数=', line.count(chr(39)), '双引号数=', line.count(chr(34)),
          '| 尾部30字符:', repr(line[-30:]))
