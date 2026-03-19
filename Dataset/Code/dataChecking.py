# -----------------------------------------------------
        # CHECK EVENT SPILT TRAIN COUNT
# -----------------------------------------------------
# # Check
# from pathlib import Path
# clips_dir = Path(r"E:\Football Dataset\Event Clips Split\train")
# for cls in clips_dir.iterdir():
#     if cls.is_dir():
#         count = len(list(cls.glob("*.mp4")))
#         print(f"  {cls.name:<25} {count}")


# -----------------------------------------------------
        # DATA TRANSFER BYTE BY BYTE
# -----------------------------------------------------

# import struct
# import tensorflow as tf
# from pathlib import Path

# OLD = Path(r"D:\Football Highlight Generation\TFRecords\train.tfrecord")
# OUTPUT = Path(r"D:\Football Highlight Generation\TFRecords\train_recovered_full.tfrecord")

# def masked_crc32c(data):
#     import crcmod
#     crc_fn = crcmod.predefined.mkCrcFun('crc-32c')
#     crc = crc_fn(data)
#     return (((crc >> 15) | (crc << 17)) + 0xa282ead8) & 0xffffffff

# def try_read_record(f, pos):
#     """Try to read a valid TFRecord at position pos. Returns (data, next_pos) or None."""
#     f.seek(pos)
#     header = f.read(8)
#     if len(header) < 8:
#         return None
#     length = struct.unpack("<Q", header)[0]

#     # Sanity check — skip insane lengths
#     if length > 50 * 1024 * 1024 or length < 10:
#         return None

#     masked_len_crc = struct.unpack("<I", f.read(4))[0]
#     data = f.read(length)
#     if len(data) < length:
#         return None
#     masked_data_crc = struct.unpack("<I", f.read(4))[0]

#     # Validate CRCs
#     if masked_crc32c(header) != masked_len_crc:
#         return None
#     if masked_crc32c(data) != masked_data_crc:
#         return None

#     return data, f.tell()


# writer = tf.io.TFRecordWriter(str(OUTPUT))
# recovered = 0
# corruption_point = 2615673354  # byte offset from error message

# with open(OLD, "rb") as f:
#     file_size = OLD.stat().st_size

#     # Phase 1: read normally up to corruption
#     pos = 0
#     while pos < corruption_point:
#         result = try_read_record(f, pos)
#         if result is None:
#             break
#         data, pos = result
#         writer.write(data)
#         recovered += 1

#     print(f"Phase 1: recovered {recovered} records before corruption")

#     # Phase 2: scan byte-by-byte after corruption to find next valid record
#     print(f"Scanning for valid records after byte {corruption_point}...")
#     scan_pos = corruption_point
#     found_next = False

#     while scan_pos < min(corruption_point + 10_000_000, file_size):  # scan 10MB window
#         result = try_read_record(f, scan_pos)
#         if result is not None:
#             print(f"  Found valid record at byte {scan_pos}")
#             found_next = True
#             data, pos = result
#             writer.write(data)
#             recovered += 1
#             break
#         scan_pos += 1
#         if scan_pos % 100000 == 0:
#             print(f"  Scanning... {scan_pos - corruption_point:,} bytes searched", end="\r")

#     # Phase 3: continue reading normally from found position
#     if found_next:
#         while pos < file_size:
#             result = try_read_record(f, pos)
#             if result is None:
#                 break
#             data, pos = result
#             writer.write(data)
#             recovered += 1

# writer.close()
# print(f"\nTotal recovered: {recovered} records")



# -----------------------------------------------------
        # GET VALID RECORDS
# -----------------------------------------------------



# import tensorflow as tf
# count = sum(1 for _ in tf.data.TFRecordDataset(str(OUTPUT)).apply(
#     tf.data.experimental.ignore_errors()
# ))
# print(f"Valid records: {count}")


# import tensorflow as tf
# from pathlib import Path

# files = [
#     Path(r"D:\Football Highlight Generation\TFRecords\train.tfrecord"),
#     Path(r"E:\Football Dataset\All records\train_new_1.tfrecord"),
#     Path(r"E:\Football Dataset\All records\train_new_2.tfrecord"),
#     Path(r"E:\Football Dataset\All records\train_new_3.tfrecord"),
#     Path(r"E:\Football Dataset\All records\train_new_4.tfrecord"),
#     Path(r"E:\Football Dataset\All records\train_new_5.tfrecord"),
#     Path(r"E:\Football Dataset\All records\train_new_6.tfrecord"),
#     Path(r"E:\Football Dataset\All records\train_new_7.tfrecord"),
#     Path(r"E:\Football Dataset\All records\train_new.tfrecord"),
# ]

# for f in files:
#     valid = 0
#     try:
#         for raw in tf.data.TFRecordDataset(str(f)):
#             valid += 1
#     except Exception as e:
#         print(f"❌ {f.name}: corrupted at record {valid} — {e}")
#         continue
#     print(f"✅ {f.name}: {valid} records")


# import tensorflow as tf
# from pathlib import Path

# OLD_TFRECORD = Path(r"D:\Football Highlight Generation\TFRecords\train.tfrecord")

# writer = tf.io.TFRecordWriter(r"D:\Football Highlight Generation\TFRecords\train_recovered.tfrecord")

# count = 0
# for raw in tf.data.TFRecordDataset(str(OLD_TFRECORD)).apply(
#     tf.data.experimental.ignore_errors()
# ):
#     writer.write(raw.numpy())
#     count += 1

# writer.close()
# print(f"Recovered {count} records")

