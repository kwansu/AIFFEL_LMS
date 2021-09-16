def create_converted_format_file(load_path, save_path):
    ''' 수작업으로 작성한 좌표정보를 포맷에 맞게 변형하여 저장한다.
    텍스트에서 줄마다 하나의 검출된 영역과 그 영역에서의 단어를 나타낸다.
    첫번째 유형은 직사각형 박스로 (x   y   w   h   word)로 기록하였다.
    두번째 유형은 사다리꼴로 (@   x1   y1   ...  x4   y4   word)로
    @로 시작하면 4개점의 x,y를 순서대로 기록하였다.
    단어 사이의 띄어쓰기가 있을수 있어서 구분은 탭(\t)으로만 하였다.
    추가로 keras-ocr에서 모두 소문자로만 나와서 문자를 소문자로 변경
    하였는데, 필요하다면 특수문자나 띄어쓰기도 처리하면 좋을 듯하다.'''

    with open(load_path, "r") as f:
        lines = f.readlines()

    with open(save_path, "w") as f:
        for line in lines:
            words = line.split('\t')
            if words[0] == '@':
                line = ' '.join(words[1:9]) + f' ##::{words[-1].lower()}'
            else:
                x, y, w, h = map(int, words[:4])
                line = f"{x} {y} {x+w} {y} {x+w} {y+h} {x} {y+h}##::{words[-1].lower()}"
            f.writelines(line)
