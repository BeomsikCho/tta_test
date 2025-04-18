import platform
import os
import matplotlib as mat
from matplotlib import font_manager

class PltPreset(object):
    @staticmethod
    def _get_times_new_roman_name():
        """
        OS에 따라 Times New Roman 폰트 경로를 잡고,
        없으면 시스템 기본 폰트로 폴백한 뒤,
        폰트 이름(내부 매타데이터)을 반환한다.
        """
        system_name = platform.system()

        if system_name == 'Darwin':  # macOS
            font_path = "/System/Library/Fonts/Times.ttc"
        elif system_name == 'Windows':  # Windows
            font_path = r"C:\Windows\Fonts\times.ttf"
        else:
            # Ubuntu 혹은 그 외 리눅스
            font_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"

        # 경로 확인
        if not os.path.exists(font_path):
            print(f"[WARN] 지정한 Times New Roman 폰트 파일이 없습니다: {font_path}")
            print("[WARN] Times New Roman 대신 시스템 기본 폰트를 사용합니다.")
            return mat.rcParams['font.family']  # 폴백
        else:
            return font_manager.FontProperties(fname=font_path).get_name()

    # staticmethod를 사용하면 클래스가 로드될 때
    # 함수 호출 가능 -> 반환값을 클래스 변수에 저장
    TIMES_NEW_ROMANS = _get_times_new_roman_name()

    @classmethod
    def base_plot_style(cls):
        mat.rcParams['font.family'] = cls.TIMES_NEW_ROMANS
        mat.rcParams['font.size'] = 15
        mat.rcParams['legend.fontsize'] = 12
        mat.rcParams['lines.linewidth'] = 2
        mat.rcParams['lines.color'] = 'r'
        mat.rcParams['axes.grid'] = True
        mat.rcParams['grid.alpha'] = 0.3
        mat.rcParams['grid.linestyle'] = '-'
        mat.rcParams['axes.xmargin'] = 0.1
        mat.rcParams['axes.ymargin'] = 0.1
        mat.rcParams["mathtext.fontset"] = "dejavuserif"
        mat.rcParams['figure.dpi'] = 500
        mat.rcParams['savefig.dpi'] = 500
        mat.rcParams['savefig.bbox'] = 'tight'
        mat.rcParams["legend.frameon"] = True
        mat.rcParams["legend.shadow"] = True      # 그림자 ON
        mat.rcParams["legend.framealpha"] = 1.0   # 범례 박스 불투명도 100%

    @classmethod
    def ijcai_plot_style(cls):
        mat.rcParams['font.family'] = cls.TIMES_NEW_ROMANS
        
        # Font Size = 9
        mat.rcParams['font.size'] = 9
        mat.rcParams['legend.fontsize'] = 9
        mat.rcParams['axes.labelsize'] = 9
        mat.rcParams['xtick.labelsize'] = 9
        mat.rcParams['ytick.labelsize'] = 9

        mat.rcParams['lines.linewidth'] = 1.0       
        mat.rcParams['lines.markersize'] = 5

        mat.rcParams['axes.grid'] = True
        mat.rcParams['grid.alpha'] = 0.3
        mat.rcParams['grid.linestyle'] = '-'

        mat.rcParams['axes.xmargin'] = 0.05
        mat.rcParams['axes.ymargin'] = 0.05

        mat.rcParams['figure.dpi'] = 300
        mat.rcParams['savefig.dpi'] = 300
        mat.rcParams['savefig.bbox'] = 'tight'
        mat.rcParams["mathtext.fontset"] = "dejavuserif"
        mat.rcParams["legend.frameon"] = True
