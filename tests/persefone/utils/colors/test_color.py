import pytest
from persefone.utils.colors.palettes import MaterialPalette
from persefone.utils.colors.color import Color


@pytest.mark.utils_colors
class TestColors(object):

    @pytest.fixture
    def augmentations_folder(self):
        return ''

    def test_palettes(self):

        palette = MaterialPalette(color_names=[])
        assert palette.size == 0, "Palette size must be 0!"
        assert palette.get_color(0) is None, "Picked color must be None!"

        material_colors = MaterialPalette.MATERIAL_COLORS
        level = '500'
        color_names = ['red', 'green', 'blue', 'teal']
        palette = MaterialPalette(color_names=color_names, color_level=level)

        assert palette.size == len(color_names), f"Palette size {palette.size} should be {len(color_names)}"

        for index, color_name in enumerate(color_names):
            color = material_colors[color_name][level]
            h = palette.get_color(index).hex
            assert h == color, f"picked color: '{h}'' should be '{color}'!"

            picked = palette.get_color(index)
            assert picked.rgb == Color.hex2rgb(color), f"picked color rgb : '{picked.rgb}' should be '{Color.hex2rgb(color)}'!"
            back = Color.rgb2hex(Color.hex2rgb(color))
            assert picked.hex == back, f"picked color hex : '{picked.rgb}'' should be '{back}'!"

            assert picked.has_alpha() is False, "Should not have alpha!"
            picked.add_alpha()
            assert picked.has_alpha() is True, "Now should have alpha!"

            rgbf_array = picked.rgbf_array
            converted = (rgbf_array * 255.).astype(int)
            assert tuple(converted) == picked.rgb, "Float coversion is wrong!"

        # COLORS
        color = Color(c="#ffaa00")
        assert color.rgb == (255, 170, 0), "Rgb color is wrong!"

        assert Color(c="#ffaa00x").rgb == (0, 0, 0), "Corrupted color must be (0,0,0)"
        assert Color(c="#dd").rgb == (0, 0, 0), "Corrupted color must be (0,0,0)"
        assert Color(c="r2d2").rgb == (0, 0, 0), "Corrupted color must be (0,0,0)"
        assert Color(c="#ffaaxx").rgb == (0, 0, 0), "Corrupted color must be (0,0,0)"

        color = Color(c=(255, 170, 0))
        assert color.hex == "#ffaa00", "Hex color is wrong!"
        assert Color(c=(128, 0, 1000)).hex == "#8000ff", "Corrupted color has to be saturated to 255"
        assert Color(c=(128, 0, 1000, 2112, 12)).hex == "#000000", "Corrupted color should be #000000"
        assert Color(c=(128, 0)).hex == "#000000", "Corrupted color should be #000000"
        assert Color(c=('hello', 0, 0)).hex == "#000000", "Corrupted color should be #000000"

        # Wrong palette
        palette = MaterialPalette(color_names=['thor_is_the_strongest_avenger'])
        assert palette.get_color(0).hex == '#000000', "Thor is not a color!"
        """
        color = p.get_color(1)

        print(color.hex)
        print(color.rgb)
        print(color.has_alpha())

        print(Color.rgb2hex(color.rgb))
        print(Color.rgb2hex(color.rgb + (255,)))

        print(Color.hex2rgb(color.hex))
        print(Color.hex2rgb(color.hex + '77'))
        color.add_alpha(a=0.5)
        print(color.hex)
        print(color.rgb)
        """
