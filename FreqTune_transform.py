import random
import numpy as np
from PIL import Image


def _preserve_dc_and_mean(fft, source_img, output_img):
    dc_index = tuple(0 for _ in range(source_img.ndim))
    fft[dc_index] = fft[dc_index].copy()

    output_real = np.real(output_img)
    output_real = output_real - np.mean(output_real) + np.mean(source_img)
    return output_real


class FreqTune_zhonly(object):
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, x):
        if random.uniform(0, 1) > self.probability:
            return x

        height = 32
        width = 32
        img = np.array(x).astype(np.uint8)
        fft_1 = np.fft.fftn(img)

        # 랜덤 영역 뽑기
        x_min = np.random.randint(width // 32, width // 2)
        x_max = np.random.randint(width // 2, width - width // 32)
        y_min = np.random.randint(height // 32, height // 2)
        y_max = np.random.randint(height // 2, height - height // 32)

        Zh_matrix = fft_1[x_min:x_max, y_min:y_max]

        # Ch 만들기
        A = 5
        a = np.random.uniform(0, A)
        Ch_matrix = np.random.uniform(-a, a, size=Zh_matrix.shape)
        # 행렬곱, 다시 넣기
        fft_1[x_min:x_max, y_min:y_max] = Zh_matrix * Ch_matrix

        img = np.fft.ifftn(fft_1)
        new_image = np.clip(img, 0, 255).astype(np.uint8)

        x = Image.fromarray(new_image)
        # x.show()
        return x

class baseline(object):
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, x):
        if random.uniform(0, 1) > self.probability:
            return x
        x.show()
        return x

class OriginalFreqTune(object):
    def __init__(self, probability=0.5, preserve_dc=False):
        self.probability = probability
        self.preserve_dc = preserve_dc

    def __call__(self, x):
        if random.uniform(0, 1) > self.probability:
            return x

        height = 32
        width = 32
        img = np.array(x).astype(np.uint8)
        fft_1 = np.fft.fftn(img)

        dc_index = tuple(0 for _ in range(fft_1.ndim))
        original_dc = fft_1[dc_index].copy() if self.preserve_dc else None

        # img pixel: matrix, make array: array
        # 랜덤 영역 뽑기
        x_min = np.random.randint(width // 32, width // 2)
        x_max = np.random.randint(width // 2, width - width // 32)
        y_min = np.random.randint(height // 32, height // 2)
        y_max = np.random.randint(height // 2, height - height // 32)
        # 중심 좌표 구하기
        matrix = fft_1[x_min:x_max, y_min:y_max]

        # 강도
        B = 0.5
        b = np.random.uniform(0, B)
        array2 = np.random.uniform(1-b, 1+b, size=fft_1.shape)

        # array2_m = np.random.uniform(-1-b, -1+b, size=fft_1.shape)
        # array2_p = np.random.uniform(1 - b, 1 + b, size=fft_1.shape)
        # array2 = np.where(np.random.rand(*fft_1.shape) > 0.5, array2_p, array2_m)

        A = 5
        a = np.random.uniform(0, A)
        array1 = np.random.uniform(-a, a, size=matrix.shape)

        # 행렬곱, 다시 넣기
        fft_1 = fft_1 * array2
        # 행렬곱, 다시 넣기
        fft_1[x_min:x_max, y_min:y_max] = matrix * array1

        if self.preserve_dc:
            fft_1[dc_index] = original_dc

        img = np.fft.ifftn(fft_1)

        # img = img.astype(np.uint8)
        # x = Image.fromarray(img)
        new_image = np.clip(img, 0, 255).astype(np.uint8)
        x = Image.fromarray(new_image)
        # x.show()
        return x


class RadialOriginalFreqTune(object):
    """FCM (OriginalFreqTune) variant with the low/high frequency split
    defined by true radial distance from DC after fftshift, instead of a
    raw FFT-index rectangle. Same fix as RadialFreqTune, applied to FCM's
    single-region formulation (one Ch region, one global Cl) instead of
    FCM-MRT's three-region one.
    """

    def __init__(self, probability=0.5, strength=1.0, preserve_dc=False):
        self.probability = probability
        self.strength = strength
        self.preserve_dc = preserve_dc

    def __call__(self, x):
        if random.uniform(0, 1) > self.probability:
            return x

        height = 32
        width = 32
        img = np.array(x).astype(np.uint8)
        fft_1 = np.fft.fftn(img)

        dc_index = tuple(0 for _ in range(fft_1.ndim))
        original_dc = fft_1[dc_index].copy() if self.preserve_dc else None

        shifted = np.fft.fftshift(fft_1, axes=(0, 1))

        yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        dist = np.sqrt((xx - width // 2) ** 2 + (yy - height // 2) ** 2)[:, :, None]
        max_r = np.sqrt((width / 2) ** 2 + (height / 2) ** 2)

        # Random low/high radius cutoff, redrawn every call -- same spirit
        # as OriginalFreqTune's randomized box size.
        r = np.random.uniform(0, max_r)
        high_mask = dist >= r

        # Cl: mild perturbation for the low-frequency core (near true DC)
        B = 0.5
        b = np.random.uniform(0, B)
        array_low = np.random.uniform(1 - b, 1 + b, size=shifted.shape)
        array_low = 1 + (array_low - 1) * self.strength

        # Ch: strong perturbation for the high-frequency outer band
        A = 5
        a = np.random.uniform(0, A)
        array_high = np.random.uniform(-a, a, size=shifted.shape)
        array_high = array_high * self.strength

        perturb = np.where(high_mask, array_high, array_low)
        shifted = shifted * perturb
        fft_1 = np.fft.ifftshift(shifted, axes=(0, 1))

        if self.preserve_dc:
            fft_1[dc_index] = original_dc

        img_out = np.fft.ifftn(fft_1)
        new_image = np.clip(np.real(img_out), 0, 255).astype(np.uint8)
        return Image.fromarray(new_image)


#GPT
class ImprovedFreqTune:
    def __init__(self, probability=0.5, A=5, B=0.5):
        self.probability = probability
        self.A = A
        self.B = B

    def __call__(self, x):
        if random.uniform(0, 1) > self.probability:
            return x

        # 이미지를 배열로 변환
        img = np.array(x).astype(np.uint8)
        height, width = img.shape[:2]

        # FFT 수행 (shift 없이)
        fft_1 = np.fft.fftn(img)

        # 랜덤으로 고주파수 영역 선택
        x_min = np.random.randint(width // 32, width // 2)
        x_max = np.random.randint(width // 2, width - width // 32)
        y_min = np.random.randint(height // 32, height // 2)
        y_max = np.random.randint(height // 2, height - height // 32)

        # 고주파수 영역 강한 변형 적용
        high_freq_region = fft_1[y_min:y_max, x_min:x_max]
        a = np.random.uniform(0, self.A)
        amplitude_perturb = np.random.uniform(-a, a, size=high_freq_region.shape)
        fft_1[y_min:y_max, x_min:x_max] = high_freq_region * (1 + amplitude_perturb)

        # 저주파수 영역 약한 변형 적용
        b = np.random.uniform(0, self.B)
        global_perturb = np.random.uniform(1 - b, 1 + b, size=fft_1.shape)
        fft_1 *= global_perturb

        # 역 FFT로 변환
        img_ifft = np.fft.ifftn(fft_1)
        new_image = np.clip(np.abs(img_ifft), 0, 255).astype(np.uint8)

        # PIL 이미지로 변환
        return Image.fromarray(new_image)


class FreqTune(object):
    def __init__(self, probability=0.5, mode='uniform', strength=1.0, preserve_dc=False):
        self.probability = probability
        self.mode = mode
        self.strength = strength
        self.preserve_dc = preserve_dc

    def __call__(self, x):
        if random.uniform(0, 1) > self.probability:
            return x

        height = 32
        width = 32
        img = np.array(x).astype(np.uint8)
        fft_1 = np.fft.fftn(img)

        dc_index = tuple(0 for _ in range(fft_1.ndim))
        original_dc = fft_1[dc_index].copy() if self.preserve_dc else None

        # img pixel: matrix, make array: array
        # 랜덤 영역 뽑기
        x_min = np.random.randint(width // 32, width // 2)
        x_max = np.random.randint(width // 2, width - width // 32)
        y_min = np.random.randint(height // 32, height // 2)
        y_max = np.random.randint(height // 2, height - height // 32)
        # 중심 좌표 구하기
        center = ((x_max-x_min) // 2, (y_max-y_min) // 2)
        matrix = fft_1[x_min:x_max, y_min:y_max]

        array1 = np.zeros(matrix.shape)
        array2 = np.zeros(fft_1.shape)

        corners_1 = [(x_min, y_min), (x_min, y_max), (x_max, y_min), (x_max, y_max)]
        corners_2 = [(0, 0), (0, 31), (31, 0), (31, 31)]
        max_distances_2 = [np.sqrt((center[0] - corner[0]) ** 2 + (center[1] - corner[1]) ** 2) for corner in corners_2]
        max_distance_2 = max(max_distances_2)

        sigma = 10
        gamma = 0.45
        # 전체를 위 center 기준으로 퍼트리기
        # 강도
        B = 0.5
        for i in range(32):  # x축 크기
            for j in range(32):  # y축 크기
                # 중심으로부터의 거리 계산
                distance = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
                # line
                value = max_distance_2 - distance
                # 선형 감소한 값을 거리로 정규화
                normalized_value = B * (value / max_distance_2)**2
                # normalized_value = B / distance

                # 배열에 값 저장
                array2[i, j] = normalized_value

                # # Gaussian
                # G_value = B * np.exp(-((distance/max_distance_2)**2) / (2 * (sigma ** 2)))
                # array2[i, j] = G_value

        # 행렬곱, 다시 넣기
        fft_1 = fft_1 * array2

        # Ch 만들기
        max_distances_1 = [np.sqrt((center[0] - corner[0]) ** 2 + (center[1] - corner[1]) ** 2) for corner in corners_1]
        max_distance_1 = max(max_distances_1)
        # max_log_1 = np.log(max_distance_1/epsilon)
        A = 5
        for i in range(x_max-x_min):  # x축 크기
            for j in range(y_max-y_min):  # y축 크기
                # 중심으로부터의 거리 계산
                distance = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
                # line
                value = max_distance_1 - distance
                # 선형 감소한 값을 거리로 정규화
                normalized_value = A * (value / max_distance_1)**2
                # 배열에 값 저장
                array1[i, j] = normalized_value

                # # Gaussian
                # G_value = A * np.exp(-((distance/max_distance_1)**2) / (2 * (sigma ** 2)))
                # array1[i, j] = G_value

        # 행렬곱, 다시 넣기
        fft_1[x_min:x_max, y_min:y_max] = matrix * array1

        if self.preserve_dc:
            fft_1[dc_index] = original_dc

        img = np.fft.ifftn(fft_1)

        # img = img.astype(np.uint8)
        # x = Image.fromarray(img)
        new_image = np.clip(img, 0, 255).astype(np.uint8)
        x = Image.fromarray(new_image)
        # x.show()
        return x

class TwoRegionFreqTune(object):
    def __init__(self, probability=0.5, mode='uniform', strength=1.0, preserve_dc=False):
        self.probability = probability
        self.mode = mode
        self.strength = strength
        self.preserve_dc = preserve_dc

    def __call__(self, x):
        if random.uniform(0, 1) > self.probability:
            return x

        height = 32
        width = 32
        img = np.array(x)
        fft_1 = np.fft.fftn(img)

        dc_index = tuple(0 for _ in range(fft_1.ndim))
        original_dc = fft_1[dc_index].copy() if self.preserve_dc else None

        # 1번 랜덤 영역 뽑기
        x_min = np.random.randint(width // 32, width // 2)
        x_max = np.random.randint(width // 2, width - width // 32)
        y_min = np.random.randint(height // 32, height // 2)
        y_max = np.random.randint(height // 2, height - height // 32)

        # 2번 랜덤 영역 뽑기
        x_min_2 = np.random.randint(width // 32, x_min + 1)
        x_max_2 = np.random.randint(x_max, width - width // 32)
        y_min_2 = np.random.randint(height // 32, y_min + 1)
        y_max_2 = np.random.randint(y_max, height - height // 32)

        # 랜덤 배열 만들기
        matrix_1 = fft_1[x_min:x_max, y_min:y_max]
        matrix_2 = fft_1[x_min_2:x_max_2, y_min_2:y_max_2]

        # 3번 배열
        B = 0.5
        b = np.random.uniform(0, B)
        array3 = np.random.uniform(1 - b, 1 + b, size=fft_1.shape)

        # 1번 배열
        A = 5
        a = np.random.uniform(0, A)
        array1 = np.random.uniform(-a, a, size=matrix_1.shape)

        # 2번 배열
        c_down = np.random.uniform(-a, 1 - b)
        c_up = np.random.uniform(1 + b, a)

        if self.mode == 'uniform':
            base_array2 = np.random.uniform(c_down, c_up, size=matrix_2.shape)
            array2 = 1 + (base_array2 - 1) * self.strength
        else:
            center_index = tuple(s // 2 for s in matrix_2.shape)
            coords = [np.arange(s) for s in matrix_2.shape]
            grids = np.meshgrid(*coords, indexing='ij')
            distances = np.sqrt(sum((g - center_index[i]) ** 2 for i, g in enumerate(grids)))
            max_distance = np.sqrt(sum((np.array(matrix_2.shape) // 2) ** 2))
            normalized_distances = distances / max_distance if max_distance > 0 else np.zeros_like(distances)

            if self.mode == 'linear':
                base_array2 = c_down + (c_up - c_down) * normalized_distances
            elif self.mode == 'log':
                k = 10
                base_array2 = c_down + (c_up - c_down) * (1 - np.log(1 + k * normalized_distances) / np.log(1 + k))
            else:
                raise ValueError('Unsupported mode: %s' % self.mode)
            array2 = 1 + (base_array2 - 1) * self.strength

        # 3번 배열 행렬곱, 다시 넣기
        array3 = 1 + (array3 - 1) * self.strength
        fft_1 = fft_1 * array3
        # 2번 배열 행렬곱, 다시 넣기
        fft_1[x_min_2:x_max_2, y_min_2:y_max_2] = matrix_2 * array2
        # 1번 배열 행렬곱, 다시 넣기
        # Ch (paper) is centered at 0 ([-a, a]), unlike Cl/Cm which are
        # centered at 1 — so strength must scale it directly, not via a
        # "1 + (x-1)*strength" blend toward 1 (that would bias it toward a
        # no-op at low strength instead of toward the paper's Ch=0 no-op).
        array1 = array1 * self.strength
        fft_1[x_min:x_max, y_min:y_max] = matrix_1 * array1

        if self.preserve_dc:
            fft_1[dc_index] = original_dc

        img_out = np.fft.ifftn(fft_1)
        new_image = np.clip(np.real(img_out), 0, 255).astype(np.uint8)
        x = Image.fromarray(new_image)
        return x


class RadialFreqTune(object):
    """FCM-MRT variant with frequency regions defined by true radial
    distance from DC after fftshift, instead of raw (unshifted) FFT-index
    rectangles. TwoRegionFreqTune's box selection is always centered on the
    raw array's middle index, which is the Nyquist bin, not DC — so its
    "regions" mix low- and high-frequency content in an uncontrolled way.
    Here, low-frequency core -> mild Cl-style perturbation, outer
    high-frequency band -> strong Ch-style perturbation, and the band
    between them -> the tunable mode (uniform/linear/log), matching the
    paper's low/mid/high-frequency framing literally.
    """

    def __init__(self, probability=0.5, mode='uniform', strength=1.0, preserve_dc=False):
        self.probability = probability
        self.mode = mode
        self.strength = strength
        self.preserve_dc = preserve_dc

    def __call__(self, x):
        if random.uniform(0, 1) > self.probability:
            return x

        height = 32
        width = 32
        img = np.array(x)
        fft_1 = np.fft.fftn(img)

        dc_index = tuple(0 for _ in range(fft_1.ndim))
        original_dc = fft_1[dc_index].copy() if self.preserve_dc else None

        # Move DC to the center along the spatial axes only (leave the color
        # channel axis untouched -- it has no spatial-frequency meaning).
        shifted = np.fft.fftshift(fft_1, axes=(0, 1))

        yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        dist = np.sqrt((xx - width // 2) ** 2 + (yy - height // 2) ** 2)[:, :, None]
        max_r = np.sqrt((width / 2) ** 2 + (height / 2) ** 2)

        # Random low/mid and mid/high radius cutoffs, redrawn every call --
        # same spirit as TwoRegionFreqTune's randomized box sizes.
        r1 = np.random.uniform(0, max_r * 0.5)
        r2 = np.random.uniform(r1, max_r)

        low_mask = dist < r1
        high_mask = dist >= r2
        mid_mask = ~low_mask & ~high_mask

        # Cl: mild perturbation for the low-frequency core (near true DC)
        B = 0.5
        b = np.random.uniform(0, B)
        array_low = np.random.uniform(1 - b, 1 + b, size=shifted.shape)
        array_low = 1 + (array_low - 1) * self.strength

        # Ch: strong perturbation for the high-frequency outer band
        A = 5
        a = np.random.uniform(0, A)
        array_high = np.random.uniform(-a, a, size=shifted.shape)
        array_high = array_high * self.strength

        # Mid band: tunable shape between Cl and Ch, by mode
        c_down = np.random.uniform(-a, 1 - b)
        c_up = np.random.uniform(1 + b, a)
        if self.mode == 'uniform':
            base_mid = np.random.uniform(c_down, c_up, size=shifted.shape)
        else:
            denom = max(r2 - r1, 1e-8)
            normalized = np.clip((dist - r1) / denom, 0, 1)
            normalized = np.broadcast_to(normalized, shifted.shape)
            if self.mode == 'linear':
                base_mid = c_down + (c_up - c_down) * normalized
            elif self.mode == 'log':
                k = 10
                base_mid = c_down + (c_up - c_down) * (1 - np.log(1 + k * normalized) / np.log(1 + k))
            else:
                raise ValueError('Unsupported mode: %s' % self.mode)
        array_mid = 1 + (base_mid - 1) * self.strength

        perturb = np.ones_like(shifted, dtype=np.float64)
        perturb = np.where(low_mask, array_low, perturb)
        perturb = np.where(mid_mask, array_mid, perturb)
        perturb = np.where(high_mask, array_high, perturb)

        shifted = shifted * perturb
        fft_1 = np.fft.ifftshift(shifted, axes=(0, 1))

        if self.preserve_dc:
            fft_1[dc_index] = original_dc

        img_out = np.fft.ifftn(fft_1)
        new_image = np.clip(np.real(img_out), 0, 255).astype(np.uint8)
        return Image.fromarray(new_image)


class BlockFreqTune(object):
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, x):
        if random.uniform(0, 1) > self.probability:
            return x
        ##
        ####nxn
        aa = [0, 8, 16, 24]
        bb = [0, 8, 16, 24]
        cc = [8, 16, 24, 32]
        dd = [8, 16, 24, 32]
        n = 4
        image_list = []
        for i in range(n):
            for j in range(n):
                imgcrop = x.crop((aa[i], bb[j], cc[i], dd[j]))
                image_list.append(imgcrop)
        ####

        result_image_list = []
        ##
        for img in image_list:
            length = 8
            img = np.array(img).astype(np.uint8)
            fft_1 = np.fft.fftn(img)

            # 랜덤 영역 뽑기
            x_min = np.random.randint(length // length, length // 2)
            x_max = np.random.randint(length // 2, length - length // length)
            y_min = np.random.randint(length // length, length // 2)
            y_max = np.random.randint(length // 2, length - length // length)

            Zh_matrix = fft_1[x_min:x_max, y_min:y_max]

            # 전체를 zl로 생각
            # Cl 만들기
            B = 0.5
            b = np.random.uniform(0, B)
            cl_matrix = np.random.uniform(1 - b, 1 + b, size=fft_1.shape)
            # 행렬곱, 다시 넣기
            fft_1 = fft_1 * cl_matrix

            # Ch 만들기
            A = 5
            a = np.random.uniform(0, A)
            Ch_matrix = np.random.uniform(-a, a, size=Zh_matrix.shape)
            # 행렬곱, 다시 넣기
            fft_1[x_min:x_max, y_min:y_max] = Zh_matrix * Ch_matrix

            newimg = np.fft.ifftn(fft_1)

            newimg = np.clip(newimg.real, 0, 255).astype(np.uint8)
            result_image_list.append(newimg)

        ####nxn
        row = []
        for ii in range(n):
            r = np.concatenate(result_image_list[ii * n : (ii + 1) * n], axis=0)
            row.append(r)
        new_image = np.concatenate(row, axis=1)
        ####

        x = Image.fromarray(new_image)
        # x.show()

        return x

class RangeChangeFreqTune(object):
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, x):
        if random.uniform(0, 1) > self.probability:
            return x

        height = 32
        width = 32
        img = np.array(x).astype(np.uint8)
        fft_1 = np.fft.fftn(img)

        # img pixel: matrix, make array: array
        # 랜덤 영역 뽑기
        x_min = np.random.randint(width // 32, width // 2)
        x_max = np.random.randint(width // 2, width - width // 32)
        y_min = np.random.randint(height // 32, height // 2)
        y_max = np.random.randint(height // 2, height - height // 32)
        # 중심 좌표 구하기
        matrix = fft_1[x_min:x_max, y_min:y_max]

        # 강도
        B = 0.5
        # b = np.random.uniform(0, B)
        # array2 = np.random.uniform(1-b, 1+b, size=fft_1.shape)
        array2 = np.random.uniform(1-B, 1+B, size=fft_1.shape)        # one uniform
        # 행렬곱, 다시 넣기
        fft_1 = fft_1 * array2

        A = 5
        # a = np.random.uniform(0, A)
        # array1 = np.random.uniform(-a, a, size=matrix.shape)
        # array1 = np.random.uniform(-A, A, size=matrix.shape)          # one uniform
        # 고주파수 범위 제외
        lower_part = np.random.uniform(-A, 1 - B, size=matrix.size // 2 + matrix.size % 2)
        upper_part = np.random.uniform(1 + B, A, size=matrix.size // 2)
        array1 = np.concatenate((lower_part, upper_part))
        np.random.shuffle(array1)
        array1 = array1.reshape(matrix.shape)
        # 행렬곱, 다시 넣기
        fft_1[x_min:x_max, y_min:y_max] = matrix * array1

        img = np.fft.ifftn(fft_1)

        # img = img.astype(np.uint8)
        # x = Image.fromarray(img)
        new_image = np.clip(img, 0, 255).astype(np.uint8)
        x = Image.fromarray(new_image)
        # x.show()
        return x


class NormalFreqtune(object):
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, x):
        if random.uniform(0, 1) > self.probability:
            return x

        height = 32
        width = 32
        img = np.array(x).astype(np.uint8)
        fft_1 = np.fft.fftn(img)

        # img pixel: matrix, make array: array
        # 랜덤 영역 뽑기
        x_min = np.random.randint(width // 32, width // 2)
        x_max = np.random.randint(width // 2, width - width // 32)
        y_min = np.random.randint(height // 32, height // 2)
        y_max = np.random.randint(height // 2, height - height // 32)
        # 중심 좌표 구하기
        matrix = fft_1[x_min:x_max, y_min:y_max]

        # sigma = 1
        # B = 0.5
        # array2 = np.random.normal(1, sigma, size=fft_1.shape) * B       ## ex1, ex2
        # array2 = np.clip(array2, 1-B, 1+B)                                  ## ex2
        #################

        sigma2 = 0.167                                                        # ex3(B=0.167), ex5(sigma2=0.167), ex6(array2에서 * B 제거)
        array2 = np.random.normal(1, sigma2, size=fft_1.shape)           ## ex3(all erase)

        # B = 0.5
        # array2 = np.random.normal(1, sigma, size=fft_1.shape)
        # array2 = np.clip(array2, 1-B, 1+B)                                ## ex4(all erase)

        # 행렬곱, 다시 넣기
        fft_1 = fft_1 * array2

        ##############
        # A = 5
        # array1 = np.random.normal(0, sigma, size=matrix.shape) * A      ## ex1, ex2
        # array1 = np.clip(array1, -A, A)                                     ## ex2
        #############

        sigma1 = 1.67                                                           # ex3(A=1.67), ex5(sigma1=1.67), ex6(array2에서 * A 제거)
        array1 = np.random.normal(0, sigma1, size=matrix.shape)             ## ex3(all erase)

        # A = 5
        # array1 = np.random.normal(0, sigma, size=matrix.shape)
        # array1 = np.clip(array1, -A, A)                                           ## ex4(all erase)

        # 행렬곱, 다시 넣기
        fft_1[x_min:x_max, y_min:y_max] = matrix * array1

        img = np.fft.ifftn(fft_1)

        # img = img.astype(np.uint8)
        # x = Image.fromarray(img)
        new_image = np.clip(img, 0, 255).astype(np.uint8)
        x = Image.fromarray(new_image)
        # x.show()
        return x
