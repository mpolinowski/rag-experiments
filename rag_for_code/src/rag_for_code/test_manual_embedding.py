from pydantic import SecretStr

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

supported_languages = [e.value for e in Language]
print(f"Supported languages :: {supported_languages}")

SOURCE_CODE = """
import copy
import os
from collections.abc import Sequence
from datetime import datetime
from typing import Any, TypeAlias

import av

os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

import cv2
import numpy as np
from cv2.typing import MatLike

cv2.utils.logging.LOG_LEVEL_SILENT

PathLike: TypeAlias = str | os.PathLike


def write_bboxes(
    background: MatLike, motion_event_list: list[list[int]], output_dir: PathLike
):
    # add bboxes to motion frame
    for bbox in motion_event_list:
        (x, y, w, h) = bbox
        cv2.rectangle(background, (x, y), (x + w, y + h), (148, 24, 248), 3)
    store_path = str(output_dir) + "/" + str(datetime.now().astimezone().timestamp())
    cv2.imwrite(store_path + ".jpg", background)


def write_alpha_channel(background: MatLike, alpha: MatLike, output_dir: PathLike):
    # bake alpha channel into motion frame
    rgba_image = cv2.cvtColor(background, cv2.COLOR_RGB2RGBA)
    cv2.mixChannels(
        (alpha,),
        (rgba_image,),
        (
            0,
            3,
        ),
    )
    store_path = str(output_dir) + "/" + str(datetime.now().astimezone().timestamp())
    cv2.imwrite(store_path + ".png", rgba_image)


def get_bbox(
    contours: Sequence[MatLike],
    motion_event_list: list[list[int]],
    logger: Any,
    debug: bool,
) -> list[list[int]]:
    min_area: int = 100
    # create bounding boxes
    for contour in contours:
        # reduce threshold for 640 px input size
        # if cv2.contourArea(contour) > 10000:
        motion_area = cv2.contourArea(contour)
        if debug:
            logger.info(f"Found motion area with size: {motion_area}.")
        if motion_area > min_area:
            (x, y, w, h) = cv2.boundingRect(contour)
            motion_event_list.append([x, y, w, h])

    if debug:
        logger.info(
            f"{len(motion_event_list)} motion areas found after filtering areas smaller than: {min_area}."
        )

    return motion_event_list


def find_motion_pixel(
    video_path: PathLike,
    output_dir: PathLike,
    frame: int,
    tries: int,
    runs: int,
    interval: int,
    motion_event_list: list[list[int]],
    logger: Any,
    debug: bool,
    save_image: bool,
    frame_returned: bool = False,
) -> tuple[list[list[int]], MatLike, bool]:

    static_background: MatLike | None = None
    first_valid_frame: MatLike | None = None

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    try:
        # Tell FFmpeg/PyAV to tolerate broken bitstreams and discard corrupt chunks
        options = {"err_detect": "ignore_err"}
        container = av.open(str(video_path), options=options)
        video_stream = container.streams.video[0]
        # Enable complete frame threading for stability
        video_stream.thread_type = "AUTO"
        video_stream.codec_context.options = {"err_detect": "ignore_err"}

        totalframes = video_stream.frames
        if totalframes == 0:
            totalframes = 999999  # Fallback if container doesn't report frame count

    except Exception as e:
        logger.error(f"Failed to open video with PyAV at {video_path}: {e}")
        return (
            motion_event_list,
            np.zeros((640, 640, 3), dtype=np.uint8),
            frame_returned,
        )

    if debug:
        logger.info(
            f"Read video file via PyAV at {video_path} and found ~{totalframes} total frames."
        )

    # Determine the maximum frame boundary or frame checking budget
    if tries == -1:
        max_frame = int(totalframes) - 100
        max_runs = None
    else:
        max_frame = min(frame + tries, int(totalframes))
        max_runs = tries

    current_frame_idx = 0
    target_frame = frame

    for packet in container.demux(video_stream):
        if packet.dts is None:
            continue

        try:
            # catch individual block corruptions
            decoded_frames = packet.decode()
        except av.error.InvalidDataError:
            if debug:
                logger.info(
                    f"Skipping corrupted packet at demux position near frame {current_frame_idx}"
                )
            continue
        except Exception:
            continue

        for raw_frame in decoded_frames:
            if current_frame_idx < target_frame:
                current_frame_idx += 1
                continue

            if (max_runs is not None and runs >= max_runs) or (
                current_frame_idx >= max_frame
            ):
                if debug:
                    logger.info("Reached frame evaluation limit or boundary.")
                break

            try:
                captured_frame = raw_frame.to_ndarray(format="bgr24")
            except Exception:
                current_frame_idx += 1
                continue

            success = captured_frame is not None
            if not success:
                target_frame += interval
                current_frame_idx += 1
                continue

            runs += 1
            frame_returned = True

            if first_valid_frame is None:
                first_valid_frame = copy.deepcopy(captured_frame)

            orig_h, orig_w = captured_frame.shape[:2]
            target_w = 640
            scale = target_w / orig_w
            target_h = int(orig_h * scale)

            gray = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2GRAY)
            gray_small = cv2.resize(
                gray, (target_w, target_h), interpolation=cv2.INTER_AREA
            )
            background = cv2.GaussianBlur(gray_small, (5, 5), 0)

            if static_background is None:
                static_background = background

            diff_frame = cv2.absdiff(static_background, background)
            thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
            thresh_frame = cv2.dilate(thresh_frame, kernel, iterations=2)
            contours, _ = cv2.findContours(
                thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            inv_scale = 1.0 / scale
            scaled_contours = [
                (contour * inv_scale).astype(np.int32) for contour in contours
            ]

            motion_event_list = get_bbox(
                scaled_contours, motion_event_list, logger, debug
            )

            if len(motion_event_list) >= 1:
                if debug:
                    logger.info(
                        f"Frame with motion found @ frame position {current_frame_idx}"
                    )
                    if save_image:
                        thresh_frame_resized = cv2.resize(
                            thresh_frame,
                            (orig_w, orig_h),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        write_alpha_channel(
                            captured_frame, thresh_frame_resized, output_dir
                        )
                        write_bboxes(captured_frame, motion_event_list, output_dir)

                container.close()
                return motion_event_list, captured_frame, frame_returned

            target_frame += interval
            current_frame_idx += 1

        if (max_runs is not None and runs >= max_runs) or (
            current_frame_idx >= max_frame
        ):
            break
            container.close()

    # Fallback return
    if first_valid_frame is None:
        logger.warning("PyAV could not read any frames from the video file!")
        return (
            motion_event_list,
            np.zeros((640, 640, 3), dtype=np.uint8),
            frame_returned,
        )
    else:
        logger.warning(
            "No motion detected, returning the first successfully read frame via PyAV."
        )
        return motion_event_list, first_valid_frame, frame_returned
"""

splitter_cpp = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size=100, chunk_overlap=0)
chunks_cpp = splitter_cpp.create_documents([SOURCE_CODE])

# DEBUG
print(type(chunks_cpp))
print(len(chunks_cpp))
print(chunks_cpp)

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
chunks = splitter.create_documents([SOURCE_CODE])

# DEBUG
print(type(chunks))
print(len(chunks))
print(chunks)


# Create vector embeddings from pages

embed_model = OpenAIEmbeddings(
    model="qwen3-embedding:0.6b",
    base_url="http://localhost:11434/v1",
    api_key=SecretStr("ollama"),
    check_embedding_ctx_length=False
)

embedded_chunks = embed_model.embed_documents([str(chunk) for chunk in chunks])

# DEBUG
print(type(embedded_chunks))
print(len(embedded_chunks))
print(embedded_chunks[0])
