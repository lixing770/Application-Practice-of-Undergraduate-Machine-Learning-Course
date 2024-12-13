import matplotlib.font_manager

# 获取系统中所有中文字体的列表
chinese_fonts = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')

# 打印中文字体名称
for font_path in chinese_fonts:
    if '中文字' in font_path:
        print(font_path)
