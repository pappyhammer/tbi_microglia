import PIL
from ScanImageTiffReader import ScanImageTiffReader
from PIL import ImageSequence
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import tifffile
from bisect import bisect_right
import scipy.ndimage


# qualitative 12 colors : http://colorbrewer2.org/?type=qualitative&scheme=Paired&n=12 + 11 diverting
BREWER_COLORS = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f',
                 '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928', '#a50026', '#d73027',
                 '#f46d43', '#fdae61', '#fee090', '#ffffbf', '#e0f3f8', '#abd9e9',
                 '#74add1', '#4575b4', '#313695']

def plot_hist_distribution(distribution_data, description, param=None, values_to_scatter=None,
                           xticks_labelsize=10, yticks_labelsize=10, x_label_font_size=15, y_label_font_size=15,
                           labels=None, scatter_shapes=None, colors=None, tight_x_range=False,
                           twice_more_bins=False, background_color="black", labels_color="white",
                           xlabel="", ylabel=None, path_results=None, save_formats="pdf",
                           v_line=None, x_range=None,
                           ax_to_use=None, color_to_use=None):
    """
    Plot a distribution in the form of an histogram, with option for adding some scatter values
    :param distribution_data:
    :param description:
    :param param:
    :param values_to_scatter:
    :param labels:
    :param scatter_shapes:
    :param colors:
    :param tight_x_range:
    :param twice_more_bins:
    :param xlabel:
    :param ylabel:
    :param save_formats:
    :return:
    """
    distribution = np.array(distribution_data)
    if len(distribution) == 0:
        print(f"Empty distribution for {description}")
        return
    if color_to_use is None:
        hist_color = "blue"
    else:
        hist_color = color_to_use
    edge_color = "white"
    if x_range is not None:
        min_range = x_range[0]
        max_range = x_range[1]
    elif tight_x_range:
        max_range = np.max(distribution)
        min_range = np.min(distribution)
    else:
        max_range = 100
        min_range = 0
    weights = (np.ones_like(distribution) / (len(distribution))) * 100
    if ax_to_use is None:
        fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                gridspec_kw={'height_ratios': [1]},
                                figsize=(12, 12))
        ax1.set_facecolor(background_color)
        fig.patch.set_facecolor(background_color)
    else:
        ax1 = ax_to_use
    bins = int(np.sqrt(len(distribution)))
    if twice_more_bins:
        bins *= 2
    hist_plt, edges_plt, patches_plt = ax1.hist(distribution, bins=bins, range=(min_range, max_range),
                                                facecolor=hist_color,
                                                edgecolor=edge_color,
                                                weights=weights, log=False, label=description)
    if values_to_scatter is not None:
        scatter_bins = np.ones(len(values_to_scatter), dtype="int16")
        scatter_bins *= -1

        for i, edge in enumerate(edges_plt):
            # print(f"i {i}, edge {edge}")
            if i >= len(hist_plt):
                # means that scatter left are on the edge of the last bin
                scatter_bins[scatter_bins == -1] = i - 1
                break

            if len(values_to_scatter[values_to_scatter <= edge]) > 0:
                if (i + 1) < len(edges_plt):
                    bool_list = values_to_scatter < edge  # edges_plt[i + 1]
                    for i_bool, bool_value in enumerate(bool_list):
                        if bool_value:
                            if scatter_bins[i_bool] == -1:
                                new_i = max(0, i - 1)
                                scatter_bins[i_bool] = new_i
                else:
                    bool_list = values_to_scatter < edge
                    for i_bool, bool_value in enumerate(bool_list):
                        if bool_value:
                            if scatter_bins[i_bool] == -1:
                                scatter_bins[i_bool] = i

        decay = np.linspace(1.1, 1.15, len(values_to_scatter))
        for i, value_to_scatter in enumerate(values_to_scatter):
            if i < len(labels):
                ax1.scatter(x=value_to_scatter, y=hist_plt[scatter_bins[i]] * decay[i], marker=scatter_shapes[i],
                            color=colors[i], s=60, zorder=20, label=labels[i])
            else:
                ax1.scatter(x=value_to_scatter, y=hist_plt[scatter_bins[i]] * decay[i], marker=scatter_shapes[i],
                            color=colors[i], s=60, zorder=20)
    y_min, y_max = ax1.get_ylim()
    if v_line is not None:
        ax1.vlines(v_line, y_min, y_max,
                   color="white", linewidth=2,
                   linestyles="dashed", zorder=5)

    ax1.legend()

    if tight_x_range:
        ax1.set_xlim(min_range, max_range)
    else:
        ax1.set_xlim(0, 100)
        xticks = np.arange(0, 110, 10)

        ax1.set_xticks(xticks)
        # sce clusters labels
        ax1.set_xticklabels(xticks)
    ax1.yaxis.set_tick_params(labelsize=xticks_labelsize)
    ax1.xaxis.set_tick_params(labelsize=yticks_labelsize)
    ax1.tick_params(axis='y', colors=labels_color)
    ax1.tick_params(axis='x', colors=labels_color)
    # TO remove the ticks but not the labels
    # ax1.xaxis.set_ticks_position('none')

    if ylabel is None:
        ax1.set_ylabel("Distribution (%)", fontsize=30, labelpad=20)
    else:
        ax1.set_ylabel(ylabel, fontsize=y_label_font_size, labelpad=20)
    ax1.set_xlabel(xlabel, fontsize=x_label_font_size, labelpad=20)

    ax1.xaxis.label.set_color(labels_color)
    ax1.yaxis.label.set_color(labels_color)

    # padding between ticks label and  label axis
    # ax1.tick_params(axis='both', which='major', pad=15)

    if ax_to_use is None:
        fig.tight_layout()
        if isinstance(save_formats, str):
            save_formats = [save_formats]
        if path_results is None:
            path_results = param.path_results
        time_str = ""
        if param is not None:
            time_str = param.time_str
        for save_format in save_formats:
            fig.savefig(f'{path_results}/{description}'
                        f'_{time_str}.{save_format}',
                        format=f"{save_format}",
                                facecolor=fig.get_facecolor())

        plt.close()


def crop_movie(centroid_cell, movie, max_width, max_height):
    """
     For given cell, get the binary mask representing this cell with the possibility to get the binary masks
     of the cells it intersects with.

    Args:
        cell:
        movie_dimensions: tuple of integers, width and height of the movie
        coord_obj: instance of
        max_width: Max width of the frame returned. Might cropped some overlaping cell if necessary
        max_height: Max height of the frame returned. Might cropped some overlaping cell if necessary
        pixels_around: how many pixels to add around the frame containing the cell and the overlapping one,
        doesn't change the mask
        buffer: How much pixels to scale the cell contour in the mask. If buffer is 0 or None, then size of the cell
        won't change.
        with_all_masks: Return a dict with all overlaps cells masks + the main cell mask. The key is an int.
     The mask consist on a binary array of with 0 for all pixels in the cell, 1 otherwise
        get_only_polygon_contour: the mask represents then only the pixels that makes the contour of the cells

    Returns: A tuple with four integers representing the corner coordinates (minx, maxx, miny, maxy)

    """
    len_frame_x = movie.shape[2]
    len_frame_y = movie.shape[1]

    # calculating the bound that will surround all the cells
    minx = None
    maxx = None
    miny = None
    maxy = None

    min_x_centroid = int(centroid_cell[0]) - int(max_height // 2)
    max_x_centroid = min_x_centroid + max_height - 1
    min_y_centroid = int(centroid_cell[1]) - int(max_width // 2)
    max_y_centroid = min_y_centroid + max_width - 1

    minx = min_x_centroid
    maxx = max_x_centroid
    miny = min_y_centroid
    maxy = max_y_centroid

    # poly_gon_cell = coord_obj.cells_polygon[cell]
    # minx_cell, miny_cell, maxx_cell, maxy_cell = np.array(list(poly_gon_cell.bounds)).astype(int)

    # trying to put the cell in, first by centering the frame around the centroid if necessary
    # then if some part of the cell still are out, we use the min of the cell
    # if min_x_centroid < minx:
    #     minx = min_x_centroid
    # # if minx_cell < minx:
    # #     minx = minx_cell
    # if min_y_centroid < miny:
    #     miny = min_y_centroid
    # # if miny_cell < miny:
    # #     miny = miny_cell
    #
    # if (minx + max_height - 1) < max_x_centroid:
    #     minx = max_x_centroid - max_height + 1
    # # if (minx + max_height - 1) < maxx_cell:
    # #     minx = maxx_cell - max_height + 1
    # if (maxy + max_width - 1) < max_y_centroid:
    #     miny = max_y_centroid - max_width + 1
    # if (miny + max_width - 1) < maxy_cell:
    #     miny = maxy_cell - max_width + 1

    # then we make sure we don't over go the border
    minx = max(0, minx)
    miny = max(0, miny)
    if minx + max_height >= len_frame_x:
        minx = len_frame_x - max_height - 1
    if miny + max_width >= len_frame_y:
        miny = len_frame_y - max_width - 1

    maxx = minx + max_height - 1
    maxy = miny + max_width - 1
    """
        old version changed the 11th of October
        #### NEW VERSION, the cell is centered ####
    
        # calculating the bound that will surround all the cells
    
        minx = int(centroid_cell[0]) - int(max_height // 2)
        miny = int(centroid_cell[1]) - int(max_width // 2)
    
        # then we make sure we don't over go the border
        minx = max(0, minx)
        miny = max(0, miny)
        if minx + max_height >= len_frame_x:
            minx = len_frame_x - max_height - 1
        if miny + max_width >= len_frame_y:
            miny = len_frame_y - max_width - 1
    
        maxx = minx + max_height - 1
        maxy = miny + max_width - 1
    """

    return movie[:, miny:maxy + 1, minx:maxx + 1]


def save_array_as_tiff(array_to_save, path_results, file_name):
    """

    :param array_to_save:
    :param path_results:
    :param file_name:
    :return:
    """
    # then saving each frame as a unique tiff
    tiff_file_name = os.path.join(path_results, file_name)
    with tifffile.TiffWriter(tiff_file_name) as tiff:
        tiff.save(array_to_save, compress=0)


def get_x_y_translation_blob(imgs):
    """
    Return from how much in x and y the blob has moved in each frame.
    Args:
        imgs:

    Returns: Return a float array of size (len(imgs) - 1) x 2

    """
    x_y_translations = []
    imgs_centroid = []
    for img_index, img in enumerate(imgs):
        centroids = find_blobs(tiff_array=img)
        if len(centroids) > 1 or len(centroids) == 0:
            # print(f"More than one blob {len(centroids)}")
            # centroids = find_blobs(tiff_array=img, with_blob_display=True)
            return []
        if img_index == 0:
            imgs_centroid.append(centroids[0])
            continue
        else:
            # measuring x_y translation
            x_y_translations.append([centroids[0][0] - imgs_centroid[-1][0], centroids[0][1] - imgs_centroid[-1][1]])
            imgs_centroid.append(centroids[0])

    # print(f"imgs_centroid {imgs_centroid}")
    # print(f"x_y_translations {np.round(x_y_translations, 2)}")
    return np.array(x_y_translations)


def binarized_frame(movie_frame, filled_value=1, percentile_threshold=90, threshold_value=None, with_uint=True,
                    keep_original_values=False):
    """
    Take a 2d-array and return a binarized version, thresholding using a percentile value.
    It could be filled with 1 or another value
    Args:
        movie_frame:
        filled_value:
        percentile_threshold:

    Returns:

    """
    img = np.copy(movie_frame)
    if threshold_value is None:
        threshold = np.percentile(img, percentile_threshold)
    else:
        threshold = threshold_value

    # print(f"in binarized_frame threshold {threshold_value}, max : {np.max(img)}")

    img[img < threshold] = 0
    if not keep_original_values:
        img[img >= threshold] = filled_value

    if with_uint:
        img = img.astype("uint8")
    else:
        img = img.astype("int8")
    return img


def max_diff_bw_frames(cell_movie, cell_id, data_id=None):
    """
    Take a movie, apply a threshold, binarize it, then make the diff between each frame but
    a x-y translation is applied between each contiguous frame in order to minimize the diff.
    Diff is actually the sum of binary pixels after substracting the previous frame to the next.
    (frames[index] - frames[index-1])
    This sum give an idea of the retraction.
    Args:
        cell_movie:

    Returns:

    """

    x_y_translation_max_range = 6
    x_y_translation_range = [0]
    for i in range(1, x_y_translation_max_range):
        x_y_translation_range.append(i)
        x_y_translation_range.append(-i)

    scale_abs_movie = np.zeros(cell_movie.shape, dtype="uint8")
    for frame_index, frame_movie in enumerate(cell_movie):
        frame_movie = cv2.normalize(frame_movie, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        # convertScaleAbs raise issues when processing 32 bit images
        # frame_movie = cv2.convertScaleAbs(frame_movie)

        # frame_movie = cv2.addWeighted(frame_movie, alpha=255, src2=0, beta=0, gamma=0, dtype=cv2.CV_8U)
        scale_abs_movie[frame_index] = frame_movie
    binary_cell_movie = np.zeros(cell_movie.shape, dtype="int8")
    # binary_diff_cell_movie = np.zeros(cell_movie.shape, dtype="int8")
    percentile_threshold = 90
    threshold_value = np.percentile(scale_abs_movie, percentile_threshold)
    high_threshold_value = np.percentile(scale_abs_movie, 98)
    # print(f"threshold_value {threshold_value}")
    for frame_index, frame_movie in enumerate(scale_abs_movie):
        # frame_movie = cv2.convertScaleAbs(frame_movie)
        # threshold_value = 200
        binary_frame = binarized_frame(movie_frame=frame_movie, filled_value=1, threshold_value=threshold_value,
                                       percentile_threshold=percentile_threshold, with_uint=False,
                                       keep_original_values=False)
        binary_cell_movie[frame_index] = binary_frame
        n_pixels_above_threshold = np.where(frame_movie >= threshold_value)[0].shape
        # if n_pixels_above_threshold[0] < 10:
        #     print(f"Instanciating binary_cell_movie {frame_index} {cell_id} {data_id}, "
        #           f"sum {np.sum(binary_cell_movie[frame_index])}, {n_pixels_above_threshold} {np.max(frame_movie)}")

        if threshold_value == 255:
            f, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(cell_movie[frame_index], cmap=cm.Greys)
            ax1.set_title(f"Original")
            ax2.imshow(frame_movie, cmap=cm.Greys)
            ax2.set_title(f"ScaleAbs")
            plt.show()
        """
        frame_movie = cv2.convertScaleAbs(frame_movie)
        # print(f"np.max(frame_movie) {np.max(frame_movie)}")
        plt.imshow(frame_movie, cmap=cm.Greys)
        plt.title(f"{frame_index + 1}")
        plt.show()
        # Applying Histogram Equalization
        # frame_movie = cv2.equalizeHist(frame_movie)

        # Denoising
        # frame_movie = cv2.fastNlMeansDenoising(frame_movie)
        # plt.imshow(frame_movie, cmap=cm.Greys)
        # plt.title(f"After denoising")
        # plt.show()
        # Gaussian Local Thresholding
        binary_frame = cv2.adaptiveThreshold(frame_movie, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                          cv2.THRESH_BINARY, 11, 0)
        plt.imshow(frame_movie, cmap=cm.Greys)
        plt.title(f"After Thresholding")
        plt.show()
        
        """
        # binary_diff_frame = binarized_frame(movie_frame=frame_movie, filled_value=1,
        #                                percentile_threshold=98, threshold_value=high_threshold_value,
        #                                     with_uint=False)
        # binary_diff_cell_movie[frame_index] = binary_diff_frame

        test_display = False
        if test_display: # or cell_id == 157:
            f, (ax1, ax2) = plt.subplots(1, 2)
            # ax1.imshow(frame_movie.astype(np.uint16), cmap=cm.Greys)

            ax1.imshow(frame_movie, cmap=cm.Greys)
            ax1.set_title(f"ScaleAbs thr {int(threshold_value)}")
            # ax1.imshow(cv2.convertScaleAbs(frame_movie), cmap=cm.Greys)
            # ax1.set_title(f"uint8 {data_id} {cell_id}")
            # print(f"threshold_value {threshold_value}")
            ax2.imshow(binary_frame, cmap=cm.Greys)
            ax2.set_title(f"Binarized")
            # tmp_frame = np.zeros(cell_movie[-1].shape, dtype="int16")
            # tmp_frame[np.where(binary_diff_frame)] = cell_movie[frame_index][np.where(binary_diff_frame)]
            # ax3.imshow(tmp_frame, cmap=cm.Greys)
            # ax3.set_title(f"{frame_index+1}")
            plt.title(f"{data_id} {cell_id} {frame_index+1}")
            plt.show()

    registered_movie = np.full(cell_movie.shape, 0, dtype="int16")
    registered_movie[0] = cell_movie[0]

    registered_binary_movie = np.full(cell_movie.shape, 0, dtype="int8")
    registered_binary_movie[0] = binary_cell_movie[0]

    x_y_translations = []

    diff_sums = []

    # with_binary_diff_cell_movie = False

    for frame_index in np.arange(1, len(cell_movie)):
        best_x_y_translation = None
        best_diff_value = None
        best_final_diff_value = None

        for x_mvt in x_y_translation_range:
            for y_mvt in x_y_translation_range:
                tmp_frame = np.zeros(cell_movie[-1].shape, dtype="int8")

                translate_frame(frame_to_translate=binary_cell_movie[frame_index],
                                frame_destination=tmp_frame,
                                x_mvt=x_mvt, y_mvt=y_mvt)
                # TODO: discard pixels at zeros due to registration
                # using np.abs to put value at -1 to 1
                diff_sum = np.sum(np.abs(tmp_frame - binary_cell_movie[frame_index - 1]))
                if False and (diff_sum == 0) and (x_mvt == x_y_translation_range[0]) and (y_mvt == x_y_translation_range[0]):
                # if  (x_mvt == x_y_translation_range[0]) and (y_mvt == x_y_translation_range[0]):
                    print(f"Plot diff sum {frame_index} {cell_id}  {data_id}")
                    f, (ax1, ax2,) = plt.subplots(1, 2)
                    # ax1.imshow(frame_movie.astype(np.uint16), cmap=cm.Greys)
                    print(f"## {frame_index} threshold_value {int(threshold_value)} sum {np.sum(binary_cell_movie[frame_index])}")
                    ax1.imshow(scale_abs_movie[frame_index], cmap=cm.Greys)
                    ax1.set_title(f"ScaleAbs thr {int(threshold_value)}")
                    ax2.imshow(binary_cell_movie[frame_index], cmap=cm.Greys)
                    ax2.set_title(f"Binarized")
                    # tmp_frame = np.zeros(cell_movie[-1].shape, dtype="int16")
                    # tmp_frame[np.where(binary_diff_frame)] = cell_movie[frame_index][np.where(binary_diff_frame)]
                    # ax3.imshow(tmp_frame, cmap=cm.Greys)
                    # ax3.set_title(f"{frame_index+1}")
                    plt.show()
                # if cell_id == 157 and frame_index == 3:
                #     print(f"diff_sum {diff_sum}")
                # we want the diff the closest from 0 at possible
                # if diff value are equals, we want to minimize the xy_translation
                if (best_diff_value is None) or (abs(best_diff_value) > abs(diff_sum)) or \
                        ((abs(best_diff_value) == abs(diff_sum)) and (np.sum(np.abs(best_x_y_translation)) >
                                                                      (np.sum(np.abs((x_mvt, y_mvt)))))):
                    # dealing with equality
                    best_diff_value = diff_sum
                    # keeping negative values for plotting the diff
                    diff_sum = np.sum(tmp_frame - binary_cell_movie[frame_index - 1])
                    # print(f'diff_sum {diff_sum}')
                    best_final_diff_value = diff_sum
                    best_x_y_translation = (x_mvt, y_mvt)

        x_mvt = best_x_y_translation[0]
        y_mvt = best_x_y_translation[1]
        translate_frame(frame_to_translate=cell_movie[frame_index],
                        frame_destination=registered_movie[frame_index],
                        x_mvt=x_mvt, y_mvt=y_mvt)

        translate_frame(frame_to_translate=binary_cell_movie[frame_index],
                        frame_destination=registered_binary_movie[frame_index],
                        x_mvt=x_mvt, y_mvt=y_mvt)

        show_best_translation = False
        if show_best_translation:
            f, (ax1, ax2, ax3) = plt.subplots(1, 3)
            # ax1.imshow(frame_movie.astype(np.uint16), cmap=cm.Greys)

            ax1.imshow(binary_cell_movie[frame_index], cmap=cm.Greys)
            ax1.set_title(f"Original")
            ax2.imshow(registered_binary_movie[frame_index], cmap=cm.Greys)
            ax2.set_title(f"Registered x {x_mvt} y {y_mvt}")
            ax3.imshow(registered_binary_movie[frame_index - 1], cmap=cm.Greys)
            ax3.set_title(f"Previous frame")
            # plt.title(f"{data_id} {cell_id} {frame_index + 1}")
            plt.show()

        # changing current frame
        translate_frame(frame_to_translate=binary_cell_movie[frame_index],
                        frame_destination=binary_cell_movie[frame_index],
                        x_mvt=x_mvt, y_mvt=y_mvt)

        # and next frame
        if frame_index < len(binary_cell_movie) - 1:
            translate_frame(frame_to_translate=binary_cell_movie[frame_index+1],
                            frame_destination=binary_cell_movie[frame_index+1],
                            x_mvt=x_mvt, y_mvt=y_mvt)
            # translate_frame(frame_to_translate=binary_diff_cell_movie[frame_index + 1],
            #                 frame_destination=binary_diff_cell_movie[frame_index + 1],
            #                 x_mvt=x_mvt, y_mvt=y_mvt)
            translate_frame(frame_to_translate=cell_movie[frame_index + 1],
                            frame_destination=cell_movie[frame_index + 1],
                            x_mvt=x_mvt, y_mvt=y_mvt)

        x_y_translations.append(best_x_y_translation)

        # print(f"best_diff_value {best_diff_value}")
        diff_sums.append(best_final_diff_value)
    # if cell_id == 157:
    #     print(f"x_y_translations {x_y_translations}")
    #     print(f"diff_sums        {diff_sums}")
    return registered_movie, registered_binary_movie, x_y_translations, diff_sums


def register_movie(cell_movie, x_y_translation):
    """

    Args:
        cell_movie: n_frames x height x width
        x_y_translation: 2d array nx2, with n == len(cell_movie) - 1

    Returns:

    """
    # TODO: Check if everything work well
    # print(f"x_y_translation {x_y_translation}")
    # first we determine the max and min x,y
    # minx = round(np.min(x_y_translation[:, 0]))
    # maxx = round(np.max(x_y_translation[:, 0]))
    # miny = round(np.min(x_y_translation[:, 1]))
    # maxy = round(np.max(x_y_translation[:, 1]))
    #
    # height = cell_movie.shape[0]
    # width = cell_movie.shape[1]
    # new_height = cell_movie.shape[0] + minx - maxx
    # new_width = cell_movie.shape[1] + miny - maxy
    # registered_movie = np.zeros((cell_movie.shape[0], new_height, new_width))
    registered_movie = np.full(cell_movie.shape, 10000, dtype="int16")
    # registered_movie[0] = cell_movie[0, -minx:width-maxx, -miny:height-maxy]
    registered_movie[0] = cell_movie[0]

    for frame_index in np.arange(1, len(cell_movie)):
        y_mvt = -1 * int(round(x_y_translation[frame_index - 1, 0]))
        x_mvt = -1 * int(round(x_y_translation[frame_index - 1, 1]))
        # print(f"x_mvt {x_mvt}, y_mvt {y_mvt}")

        translate_frame(frame_to_translate=cell_movie[frame_index],
                        frame_destination=registered_movie[frame_index],
                        x_mvt=x_mvt, y_mvt=y_mvt)

        # if x_mvt == 0 and y_mvt == 0:
        #     registered_movie[frame_index] = cell_movie[frame_index]
        # elif x_mvt < 0 and y_mvt == 0:
        #     registered_movie[frame_index, :x_mvt, :] = cell_movie[frame_index, -x_mvt:, :]
        # elif x_mvt > 0 and y_mvt == 0:
        #     registered_movie[frame_index, x_mvt:, :] = cell_movie[frame_index, :-x_mvt, :]
        # elif y_mvt < 0 and x_mvt == 0:
        #     registered_movie[frame_index, :, :y_mvt] = cell_movie[frame_index, :, -y_mvt:]
        # elif y_mvt > 0 and x_mvt == 0:
        #     registered_movie[frame_index, :, y_mvt:] = cell_movie[frame_index, :, :-y_mvt]
        # elif x_mvt < 0 and y_mvt < 0:
        #     registered_movie[frame_index, :x_mvt, :y_mvt] = cell_movie[frame_index, -x_mvt:,  -y_mvt:]
        # elif x_mvt > 0 and y_mvt > 0:
        #     registered_movie[frame_index, x_mvt:, y_mvt:] = cell_movie[frame_index, :-x_mvt, :-y_mvt]
        # elif x_mvt > 0 and y_mvt < 0:
        #     registered_movie[frame_index, x_mvt:, :y_mvt] = cell_movie[frame_index, :-x_mvt, -y_mvt:]
        # elif x_mvt < 0 and y_mvt > 0:
        #     registered_movie[frame_index, :x_mvt, y_mvt:] = cell_movie[frame_index, -x_mvt:, :-y_mvt]

    return registered_movie


def translate_frame(frame_to_translate, frame_destination, x_mvt, y_mvt):
    """
    Fill frame_destination with frame_to_translate after applying x_mvt and y_mvt translation
    Args:
        frame_to_translate:
        frame_destination:
        x_mvt:
        y_mvt:

    Returns:

    """
    scipy.ndimage.shift(frame_to_translate, shift=(x_mvt, y_mvt), output=frame_destination)
    # if x_mvt == 0 and y_mvt == 0:
    #     frame_destination[:] = frame_to_translate
    # elif x_mvt < 0 and y_mvt == 0:
    #     frame_destination[:x_mvt, :] = frame_to_translate[-x_mvt:, :]
    # elif x_mvt > 0 and y_mvt == 0:
    #     frame_destination[x_mvt:, :] = frame_to_translate[:-x_mvt, :]
    # elif y_mvt < 0 and x_mvt == 0:
    #     frame_destination[:, :y_mvt] = frame_to_translate[:, -y_mvt:]
    # elif y_mvt > 0 and x_mvt == 0:
    #     frame_destination[:, y_mvt:] = frame_to_translate[:, :-y_mvt]
    # elif x_mvt < 0 and y_mvt < 0:
    #     frame_destination[:x_mvt, :y_mvt] = frame_to_translate[-x_mvt:, -y_mvt:]
    # elif x_mvt > 0 and y_mvt > 0:
    #     frame_destination[x_mvt:, y_mvt:] = frame_to_translate[:-x_mvt, :-y_mvt]
    # elif x_mvt > 0 and y_mvt < 0:
    #     frame_destination[x_mvt:, :y_mvt] = frame_to_translate[:-x_mvt, -y_mvt:]
    # elif x_mvt < 0 and y_mvt > 0:
    #     frame_destination[:x_mvt, y_mvt:] = frame_to_translate[-x_mvt:, :-y_mvt]


def load_tiff_movie(tiff_file_name):
    """
    Load a tiff movie from tiff file name.
    Args:
        tiff_file_name:

    Returns: a 3d array: n_frames * width_FOV * height_FOV

    """
    try:
        # start_time = time.time()
        tiff_movie = ScanImageTiffReader(tiff_file_name).data()
        # stop_time = time.time()
        # print(f"Time for loading movie with ScanImageTiffReader: "
        #       f"{np.round(stop_time - start_time, 3)} s")
    except Exception as e:
        im = PIL.Image.open(tiff_file_name)
        n_frames = len(list(ImageSequence.Iterator(im)))
        dim_y, dim_x = np.array(im).shape
        tiff_movie = np.zeros((n_frames, dim_y, dim_x), dtype="uint16")
        for frame, page in enumerate(ImageSequence.Iterator(im)):
            tiff_movie[frame] = np.array(page)
    return tiff_movie


def find_blobs(tiff_array, results_path=None, result_id=None, with_blob_display=False):
    """
    Blobs here will correspond to cells
    :param tiff_array: (np.array)
    :param results_path:
    :param result_id:
    :param with_blob_display: (bool) if True, the blob outcome will be save in a figure
    :return:
    """
    # img = ((tiff_array - tiff_array.min()) * (1 / (tiff_array.max() - tiff_array.min()) * 255)).astype('uint8')
    img = binarized_frame(movie_frame=tiff_array, filled_value=255, percentile_threshold=98)
    # img = tiff_array.copy()
    # # plt.imshow(img)
    # # plt.show()
    # threshold = np.percentile(img, 98)
    #
    # img[img < threshold] = 0
    # img[img >= threshold] = 255
    # # print(tiff_array.shape)
    #
    # # print(f"im min {np.min(tiff_array[0])} max {np.max(tiff_array[0])}")
    # img = img.astype("uint8")

    # img = np.reshape(img, (img.shape[0], img.shape[1], 1))
    # print(img.shape)
    # Detect blobs.

    # invert image (blob detection only works with white background)
    img = cv2.bitwise_not(img)

    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    # params.minThreshold = 10
    # params.maxThreshold = 200

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 25

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.3

    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.01

    # plt.imshow(tiff_array[0], cmap=cm.Reds)
    # plt.show()
    # Set up the detector with default parameters.
    ## check opencv version and construct the detector
    is_v2 = cv2.__version__.startswith("2.")
    if is_v2:
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(img)
    # print(f"len keypoints {len(keypoints)}")

    # Draw detected blobs as red circles.
    if with_blob_display:
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Show keypoints
        # cv2.imshow("Keypoints", im_with_keypoints)
        cv2.imwrite(os.path.join(results_path, f"{result_id}_blobs.png"), im_with_keypoints)
        cv2.waitKey(0)

    from shapely.geometry import Polygon

    centers_of_gravity = [(keypoint.pt[0], keypoint.pt[1]) for keypoint in keypoints]
    # for keypoint in keypoints:
    #     x = keypoint.pt[0]
    #     y = keypoint.pt[1]
    #     s = keypoint.size
    #
    #     # polygon = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
    #     # polygon.centroid
    #     print(f"x {x}, y {y}")
    return centers_of_gravity


def analyze_movie(tiff_file_name, results_id, results_path):
    """
    Main analysis
    :param tiff_file_name: (str)
    :param results_id:  (str)
    :param results_path: (str)
    :return:
    """
    movie = load_tiff_movie(tiff_file_name=tiff_file_name)
    print(movie.shape)

    new_results_path = os.path.join(results_path, results_id)
    if not os.path.exists(new_results_path):
        os.mkdir(new_results_path)

    # each element if a tuple of 2 int represent the center of a cell
    # centers_of_gravity = find_blobs(movie.copy()[0])
    centers_of_gravity = find_blobs(np.mean(movie, axis=0), with_blob_display=True,
                                    results_path=new_results_path, result_id=results_id)

    print(f"Nb of blobs: {len(centers_of_gravity)}")
    # raise Exception("TOTO")
    # list of 3d array reprensenting a crop movie centered on a blob (cell)
    cells_movie = []
    # now we want to build as many arrays as centers_of_gravity
    for cell_index, centroid_cell in enumerate(centers_of_gravity):
        cropped_movie = crop_movie(centroid_cell=centroid_cell, movie=movie, max_width=50, max_height=50)
        cells_movie.append(cropped_movie)
        # save_array_as_tiff(array_to_save=cropped_movie, path_results=results_path, file_name=f"{cell_index}_{results_id}.tiff")

    # list of float representing the difference of pixels intensity between a frame and the previous one.
    # if the difference is negative, it should mean there was some contraction, if it is positive, it should
    # mean there was some expension.
    all_diff_sums = []
    for cell_movie_index, cell_movie in enumerate(cells_movie):
        try_second_version = True
        if try_second_version:
            # print(f"{cell_movie_index}_{results_id}")
            registered_movie, registered_binary_movie, \
            x_y_translations, diff_sums = max_diff_bw_frames(cell_movie.copy(),
                                                             cell_id=cell_movie_index,
                                                             data_id=results_id)
            all_diff_sums.extend(diff_sums)
            # we save the crop movie, the version without added registration (motion correction)
            # and the version with motion correction
            save_array_as_tiff(array_to_save=registered_movie, path_results=new_results_path,
                               file_name=f"{cell_movie_index}_new_{results_id}.tiff")
            save_array_as_tiff(array_to_save=cell_movie, path_results=new_results_path,
                               file_name=f"{cell_movie_index}_{results_id}.tiff")
        else:
            x_y_translation = get_x_y_translation_blob(cell_movie)
            if len(x_y_translation) == 0:
                new_cell_movie = cell_movie
            else:
                new_cell_movie = register_movie(cell_movie, x_y_translation)
                save_array_as_tiff(array_to_save=new_cell_movie, path_results=new_results_path,
                                   file_name=f"{cell_movie_index}_new_{results_id}.tiff")
                save_array_as_tiff(array_to_save=cell_movie, path_results=new_results_path,
                                   file_name=f"{cell_movie_index}_{results_id}.tiff")

    # TODO: Doing the diff between the different images,
    #  take the minimum diff after trying a few sliding over the window
    # print(f"all_diff_sums {all_diff_sums}")
    plot_hist_distribution(distribution_data=all_diff_sums,
                           description=f"hist_prediction_distribution_diffs_{results_id}",
                           path_results=results_path,
                           tight_x_range=True,
                           twice_more_bins=True,
                           xlabel=f"Diffs",
                           save_formats="png")
    np.save(os.path.join(results_path, results_id + ".npy"), all_diff_sums)

def plot_cdf_from_distributions(distributions_dict, results_path, colors_dict, linestyle_dict, len_dict_key=8, title=""):
    # conditions = ['H', 'N', 'F']
    # conditions_color = {'H': "red", 'N': "blue", 'F': "black"}
    # distributions_dict = dict()
    all_values = None
    for key, distribution in distributions_dict.items():
        if all_values is None:
            all_values = distribution
        else:
            all_values = np.concatenate((all_values, distribution))
    all_values = np.unique(all_values)
    bin_size = 1
    intensity_intervals = np.arange(int(np.min(all_values)), np.ceil(np.max(all_values)) + 1, bin_size)

    cdf_dict = dict()
    for key, distribution in distributions_dict.items():
        cdf_dict[key] = np.zeros(len(intensity_intervals))
        values, counts = np.unique(distribution, return_counts=True)
        for value, count in zip(values, counts):
            interval_index = bisect_right(intensity_intervals, value) - 1
            cdf_dict[key][interval_index] = count
        cdf_dict[key] = (cdf_dict[key] / np.sum(cdf_dict[key])) * 100
        # now sum should be equal to 100
        cdf_dict[key] = np.cumsum(cdf_dict[key])

    # for index, condition_1 in enumerate(conditions[:-1]):
    #     for condition_2 in conditions[index + 1:]:
    #         d_ks, p_value = scipy.stats.ks_2samp(distributions_dict[condition_1], distributions_dict[condition_2])
    #         print(f"{condition_1} vs {condition_2}: D={np.round(d_ks, 3)}, p={p_value}")

    fig, ax1 = plt.subplots(nrows=1, ncols=1,
                            figsize=(8, 8))

    index = 0
    for key, cdf in cdf_dict.items():
        label_str = key
        if label_str[len_dict_key:] in colors_dict:
            ax1.plot(intensity_intervals, cdf, c=colors_dict[label_str[len_dict_key:]], linewidth=1.5,
                     ls=linestyle_dict[label_str[:len_dict_key]], label=label_str)
        index += 1

    plt.legend()
    # chartBox = ax1.get_position()
    # ax1.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.6, chartBox.height])
    # ax1.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)
    plt.xlabel("Pixel intensity", fontweight="bold", fontsize=12, labelpad=10)
    # ax1.set_xlim(-500, 500)
    # ax1.set_xscale("log")
    plt.ylabel("Cumulative percent", fontweight="bold", fontsize=12, labelpad=10)
    time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")

    fig.savefig(f'{results_path}/cdf_{title}{time_str}.pdf',
                format="pdf")
    plt.close()

def find_dir_with_keywords(path_to_explore, keyword):
    """
    Case insensitive
    :param path_to_explore:
    :param keyword:
    :return:
    """
    dir_names = []
    # look for filenames in the fisrst directory, if we don't break, it will go through all directories
    for (dirpath, dirnames, local_filenames) in os.walk(path_to_explore):
        dir_names.extend(dirnames)
        break
    dir_names = [d for d in dir_names if keyword.lower() in d.lower()]

    return dir_names

if __name__ == '__main__':
    # root_path = "/Users/pappyhammer/Documents/academique/these_inmed/tbi_microglia_github/"
    first_data_version = False
    # root_path = "/media/julien/Not_today/tbi_microglia/"
    root_path = "/Volumes/JD/these_inmed/tbi_microglia/"
    if first_data_version:
        data_path = os.path.join(root_path, "data/")
    else:
        data_path = os.path.join(root_path, "data/Microglia in vivo/")

    results_path = os.path.join(root_path, "results")
    time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")

    results_path = os.path.join(results_path, time_str)
    os.mkdir(results_path)

    # if use_pre_computed_data is set to True, it means we use the data pre computed to plot the figures
    # (cumulative distribution plots)
    # if False, then we apply to algorithm to split the movie in small patch aroudn cell and compute
    # the shift to apply registration and then compute the pixel intensity difference between frames
    use_pre_computed_data = True
    if first_data_version:
        all_mice = ["Mouse 64", "Mouse 66"]
        len_dict_key = 8
    else:
        all_mice = ["M1", "M2", "M3", "M4", "M6"]
        len_dict_key = 2

    precomputed_data_dir = os.path.join(root_path, "distributions")
    if use_pre_computed_data:
        """
        '-'	solid line style
        '--'	dashed line style
        '-.'	dash-dot line style
        ':'	dotted line style
        """
        linestyles = ['-', '--', '-.', ':', '-']
        # we do pair of all_mice
        pairs_of_mice = [(all_mice[i], all_mice[j]) for i in range(len(all_mice))
                          for j in range(i + 1, len(all_mice))]
        # adding plots of each mouse separatly
        pairs_of_mice.extend([[all_mice[i]] for i in range(len(all_mice))])
        colors_dict = dict()
        for mice in pairs_of_mice:
            distributions_dict = dict()
            linestyle_dict = dict()
            for mouse_index, mouse in enumerate(mice):
                print(f"## {mouse}")
                linestyle_dict[mouse] = linestyles[mouse_index]
                file_names = []
                # look for filenames in the fisrst directory, if we don't break, it will go through all directories
                for (dirpath, dirnames, local_filenames) in os.walk(precomputed_data_dir):
                    file_names.extend(local_filenames)
                    break
                file_names = [f for f in file_names if f.startswith(mouse)]
                fill_colors_dict = len(colors_dict) == 0
                for file_name_index, file_name in enumerate(file_names):
                    distribution = np.load(os.path.join(precomputed_data_dir, file_name))
                    # session_id = file_name[len(mouse)+1:-4]
                    session_id = file_name[:-4]
                    index_avg = session_id.index("AVG")
                    session_id = session_id[:index_avg - 1]
                    if fill_colors_dict:
                        # 8 is the length of the mouse id
                        colors_dict[session_id[len_dict_key:]] = BREWER_COLORS[file_name_index % len(BREWER_COLORS)]
                    pos_values = np.sum(distribution[distribution > 0])
                    neg_values = np.sum(np.abs(distribution[distribution < 0]))

                    if pos_values > neg_values:
                        print(f"{session_id} > 0: {np.round(pos_values, 2)} vs {-1*np.round(neg_values, 2)}")
                    else:
                        print(f"{session_id} < 0: {-1*np.round(neg_values, 2)} vs {np.round(pos_values, 2)}")

                    distributions_dict[session_id] = distribution
            # BREWER_COLORS[index % len(BREWER_COLORS)]
            plot_cdf_from_distributions(distributions_dict, results_path, colors_dict=colors_dict,
                                        linestyle_dict=linestyle_dict, title=f"{'_'.join(mice)}_",
                                        len_dict_key=len_dict_key)
    else:
        # results_id = "XYCTZ_Substack_9-16"
        # tiff_file_name = os.path.join(data_path, "registered [XYCTZ] Substack (9-16).tif")
        if first_data_version:
            conditions = ["1d post injury", "Baseline", "Injury"]
            subfolders = {"Injury": ["Before injury", "After injury"],
                          "Baseline": ["Before zoom", "After zoom"],
                          "1d post injury": [""]}
        else:
            conditions = ["Injury day", "1d post injury", "3d post injury", "5d post injury"]
            subfolders = {"Injury day": ["1h post injury", "30min baseline"],
                          "1d post injury": [""],
                          "3d post injury": [""],
                          "5d post injury": [""]}

        for mouse in all_mice:
            mouse_path = find_dir_with_keywords(data_path, mouse)
            if len(mouse_path) == 0:
                raise Exception(f"{mouse} not found in {data_path}")
            else:
                mouse_path = os.path.join(data_path, mouse_path[0])
            for condition in conditions:
                condition_path = find_dir_with_keywords(mouse_path, condition)
                if len(condition_path) == 0:
                    # raise Exception(f"{condition} not found in {mouse_path}")
                    print(f"{condition} not found in {mouse_path}")
                    continue
                else:
                    condition_path = os.path.join(mouse_path, condition_path[0])
                for subfolder in subfolders[condition]:
                    if subfolder == "":
                        current_data_path = condition_path
                    else:
                        current_data_path = find_dir_with_keywords(condition_path, subfolder)
                        if len(current_data_path) == 0:
                            raise Exception(f"{subfolder} not found in {condition_path}")
                        else:
                            current_data_path = os.path.join(condition_path, current_data_path[0])

                    # then we look for all tif files starting with a f
                    file_names = []
                    # look for filenames in the fisrst directory, if we don't break, it will go through all directories
                    for (dirpath, dirnames, local_filenames) in os.walk(current_data_path):
                        file_names.extend(local_filenames)
                        break

                    # if first_data_version:
                    file_names = [f for f in file_names if "AVG" in f and (f.endswith(".tif") or f.endswith(".tiff"))]
                    for tiff_file_name in file_names:
                        if tiff_file_name.endswith(".tif"):
                            results_id = mouse + "_" + condition + "_" + subfolder + "_" + tiff_file_name[:-4]
                        else:
                            # ends with .tiff
                            results_id = mouse + "_" + condition + "_" + subfolder + "_" + tiff_file_name[:-5]
                        print(f"## Analyzing  {results_id}")
                        analyze_movie(tiff_file_name=os.path.join(current_data_path, tiff_file_name),
                                      results_id=results_id, results_path=results_path)
                        print("")



