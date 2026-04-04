import json

with open('/Users/lostgreen/Downloads/大文件/2_3_m_youtube_oe_v0_1_qa_processed.json', 'r') as f:
    first_line = json.loads(f.readline())
    s = set()
    for i in range(1000):
        s.add(first_line[i]["id"])

for x in s:
    print("https://www.youtube.com/watch?v=" + x)