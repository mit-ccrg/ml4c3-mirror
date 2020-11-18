# Imports: standard library
import os
import time
import base64
import struct
import argparse
import warnings

# Imports: third party
import h5py
import pandas as pd
from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, padding
from cryptography.hazmat.primitives.hmac import HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, modes, algorithms

# pylint: disable=invalid-name


class NameEncrypter:
    """
    Class to encrypt and decrypt names such as EDW folders and hd5 filenames
    and groups.
    """

    def __init__(self, path_key: str = "/media/ml4c3/tensorization.key"):
        """
        Init Name Encrypter.

        :param path_key: <str> Path to the key.
        """
        self.path_key = path_key
        self.key = None

    def create_key(self, overwrite: bool = False):
        """
        Generate key for encrypting and decrypting.

        :param overwrite: <bool> Flag to overwrite the key.
        """
        if os.path.exists(self.path_key) and (not overwrite):
            raise OSError(
                "Key already exists. If replaced, decryption cannot be "
                "performed. Set overwrite=True if you want to force the "
                "generation of a new one.",
            )

        key = Fernet.generate_key()
        with open(self.path_key, "wb") as key_file:
            key_file.write(key)

    def load_key(self):
        """
        Load key.
        """
        self.key = open(self.path_key, "rb").read()

    def encrypt_names(self, name: str) -> str:
        """
        Encrypt string.

        :param name: <str> String to be encrypted.
        :return: <str> Encrypted name.
        """
        # Check key has been loaded
        if not self.key:
            raise ValueError("Key not loaded. Please load key before encrypting.")

        # Encrypt name
        encoded_name = name.encode()
        # Using modified encrypting method (instead of Fernet encrypt method)
        iv = b"icd\xe3/]\xb2\xac\xa76\xa3\xacY$\xf1\xa1"  # Constant iv
        current_value = 100  # Constant value
        encrypted_name = self._encrypting_message(encoded_name, current_value, iv)
        return encrypted_name.decode()

    def _encrypting_message(self, data, current_value, iv):
        # Get key
        key = base64.urlsafe_b64decode(self.key)
        # Pad data
        padder = padding.PKCS7(algorithms.AES.block_size).padder()
        padded_data = padder.update(data) + padder.finalize()
        # Get encryptor
        encryptor = Cipher(
            algorithms.AES(key[16:]),
            modes.CBC(iv),
            default_backend(),
        ).encryptor()
        # Encrypt text
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()
        basic_parts = b"\x80" + struct.pack(">Q", current_value) + iv + ciphertext
        h = HMAC(key[:16], hashes.SHA256(), backend=default_backend())
        h.update(basic_parts)
        hmac = h.finalize()
        return base64.urlsafe_b64encode(basic_parts + hmac)

    def decrypt_names(self, name: str) -> str:
        """
        Decrypt string.

        :param name: <str> String to be decrypted.
        :return: <str> Decrypted name.
        """
        # Check key has been loaded
        if not self.key:
            raise ValueError("Key not loaded. Please load key before decrypting.")

        # Decrypt name
        encoded_name = name.encode()
        # Using regular decrypting method
        encryptor = Fernet(self.key)
        decrypted_name = encryptor.decrypt(encoded_name)
        return decrypted_name.decode()

    def encrypt_folders(
        self,
        edw_dir: str = "/media/ml4c3/edw",
        decrypt: bool = False,
    ):
        """
        Encrypt folder names.

        :param edw_dir: <str> Directory where the edw data is stored.
        :param decrypt: <bool> Flag if decrypting instead of encrypting.
        """
        list_folders = next(os.walk(edw_dir))[1]
        for folder in list_folders:
            folder_dir = os.path.join(edw_dir, folder)
            subfolders = next(os.walk(folder_dir))[1]
            for subfolder in subfolders:
                subfolder_dir = os.path.join(folder_dir, subfolder)
                if decrypt:
                    changed_subfolder = self.decrypt_names(subfolder)
                else:
                    changed_subfolder = self.encrypt_names(subfolder)
                os.rename(subfolder_dir, os.path.join(folder_dir, changed_subfolder))
            if decrypt:
                changed_folder = self.decrypt_names(folder)
            else:
                changed_folder = self.encrypt_names(folder)
            os.rename(folder_dir, os.path.join(edw_dir, changed_folder))

    def encrypt_csv_file(
        self,
        csv_file: str = "/media/ml4c3/cohorts_lists/xref_file.csv",
        decrypt: bool = False,
    ):
        """
        Encrypt csv file contents (only PHI: MRN and CSN).

        :param csv_file: <str> .CSV file containing the MRNs and CSNs.
        :param decrypt: <bool> Flag if decrypting instead of encrypting.
        """
        df_csv = pd.read_csv(csv_file, dtype=str)

        for idx, row in df_csv.iterrows():
            mrn = row["MRN"]
            csn = row["PatientEncounterID"]
            if decrypt:
                mrn_changed = self.decrypt_names(mrn)
                csn_changed = self.decrypt_names(csn)
            else:
                mrn_changed = self.encrypt_names(mrn)
                csn_changed = self.encrypt_names(csn)
            df_csv.at[idx, "MRN"] = mrn_changed
            df_csv.at[idx, "PatientEncounterID"] = csn_changed
        df_csv.to_csv(csv_file, index=False)

    def encrypt_hd5(self, hd5_dir: str = "/media/ml4c3/hd5", decrypt: bool = False):
        """
        Decrypt hd5 name and csn groups.

        :param hd5_dir: <str> Directory where the hd5 files are stored.
        :param decrypt: <bool> Flag if decrypting instead of encrypting.
        """
        hd5_files = sorted(os.listdir(hd5_dir))
        if not all(file.endswith(".hd5") for file in hd5_files):
            warnings.warn("Not all listed files end with .hd5 extension")
            hd5_files = [file for file in hd5_files if file.endswith(".hd5")]
        for file in hd5_files:
            file_dir = os.path.join(hd5_dir, file)
            with h5py.File(file_dir, "r+") as f1:
                for database in f1.keys():
                    for visit_id in f1[database].keys():
                        if decrypt:
                            visit_id_changed = self.decrypt_names(visit_id)
                        else:
                            visit_id_changed = self.encrypt_names(visit_id)
                        f1[database][visit_id_changed] = f1[database][visit_id]
                        del f1[database][visit_id]
            if decrypt:
                file_changed = self.decrypt_names(file[:-4])
            else:
                file_changed = self.encrypt_names(file[:-4])
            os.rename(file_dir, os.path.join(hd5_dir, file_changed + ".hd5"))


def parse_arguments():
    parser = argparse.ArgumentParser(description="Encrypt folders and files.")
    parser.add_argument(
        "--path_edw",
        type=str,
        default="/media/ml4c3/edw/",
        help="Directory with EDW .csv files.",
    )
    parser.add_argument(
        "--tensors",
        type=str,
        default="/media/ml4c3/hd5/",
        help="Directory with tensorized .hd5 tensors.",
    )
    parser.add_argument(
        "--path_xref",
        type=str,
        default="/media/ml4c3/xref.csv",
        help="Full path of the file where EDW and Bedmaster "
        "are cross referenced. CSV file which indicates the "
        "corresponding MRN and CSN of each Bedmaster file.",
    )
    parser.add_argument(
        "--path_key",
        type=str,
        default="~/icu/tensorization.key",
        help="Full path to encryption key.",
    )
    parser.add_argument(
        "--encryption_type",
        type=str,
        default="folder",
        help="Types can be: 'folder', 'csv_file' and 'hd5_file'.",
    )
    parser.add_argument(
        "--decrypt_flag",
        action="store_true",
        help="If parameter set, files and folders are decrypted instead.",
    )
    parser.add_argument(
        "--keep_key",
        action="store_true",
        help="If parameter set, previous key is maintained.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace):
    encrypter = NameEncrypter(args.path_key)

    if not args.keep_key:
        encrypter.create_key()
    encrypter.load_key()

    if args.encryption_type == "folder":
        init_time = time.time()
        encrypter.encrypt_folders(args.path_edw, decrypt=args.decrypt_flag)
        elapsed_time = time.time() - init_time
        print(f"Folders encryption time: {round(elapsed_time, 4)} seconds.")
    elif args.encryption_type == "csv_file":
        init_time = time.time()
        encrypter.encrypt_csv_file(args.path_xref, decrypt=args.decrypt_flag)
        elapsed_time = time.time() - init_time
        print(f"CSV file encryption time: {round(elapsed_time, 4)} seconds.")
    elif args.encryption_type == "hd5_file":
        init_time = time.time()
        encrypter.encrypt_hd5(args.path_hd5, decrypt=args.decrypt_flag)
        elapsed_time = time.time() - init_time
        print(f"HD5 encryption time: {round(elapsed_time, 4)} seconds.")
    else:
        raise ValueError(
            f"Name '{args.encryption_type}' for ENCRYPTION_TYPE argument does not"
            f"exist. Possible types are: 'folder', 'csv_file' and 'hd5_file'.",
        )


if __name__ == "__main__":
    args = parse_arguments()
    run(args)
