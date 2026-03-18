import os
import numpy as np
from Bio import SeqIO
from collections import defaultdict
from copy import deepcopy
import pandas as pd
import logging
import subprocess
from src.core.base_feature import BaseFeatureExtractor
from src.utils.logger import get_logger
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Any
import multiprocessing

# 配置日志
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = get_logger(__name__)
# logger.addHandler(logging.NullHandler())

class GeneContentExtractor(BaseFeatureExtractor):
    """
    基因组特征提取器：基于基因组的编码序列 (CDS) 计算基因组特征
    特征包括：
    - 单核苷酸频率
    - 双核苷酸频率及 Odds Ratio
    - 密码子使用偏好
    - 氨基酸频率
    """
    def __init__(self, config):
        super().__init__(config)
        self.threads = os.cpu_count()
    def load(self, file_path):
        """读取特征文件"""
        df = pd.read_parquet(file_path)
        return df
    def generate(self, fasta_path, output_path):
        """基于fasta生成特征"""
        # print(f"Generating genomic traits features for {fasta_path}...")

        # 假设 CDS 文件路径是基于 fasta_path 推导出来的
        cds_path = fasta_path.rsplit('.', 1)[0] + f'_{self.config["paths"]["source_suffix"]["cds"]}'
        prot_path = fasta_path.rsplit('.', 1)[0] + f'_{self.config["paths"]["source_suffix"]["protein"]}'

        if not os.path.exists(cds_path):
            logger.info(f"      CDS file not found at: {cds_path}. Run gene prediction first.")
            command = [
                'python', self.prodigal_path,
                '-t', str(self.config['general']['threads']),
                '-q',
                '-i', fasta_path,
                '-d', cds_path,
                '-a', prot_path,
                # '-p', 'meta'
            ]
            subprocess.run(command,stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        encoder = GenomicTraitsEncoder(fasta_path, cds_path)
        df_features = encoder.encode()
        df_features.to_parquet(output_path)


class GenomicConstants:
    """存储生物学常数和模板"""
    CODON_TABLE = {
        "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
        "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R", "AGA": "R", "AGG": "R",
        "AAT": "N", "AAC": "N", "GAT": "D", "GAC": "D", "TGT": "C", "TGC": "C",
        "CAA": "Q", "CAG": "Q", "GAA": "E", "GAG": "E", "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
        "CAT": "H", "CAC": "H", "ATG": "M", "ATT": "I", "ATC": "I", "ATA": "I",
        "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L", "TTA": "L", "TTG": "L",
        "AAA": "K", "AAG": "K", "TTT": "F", "TTC": "F", "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
        "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S", "AGT": "S", "AGC": "S",
        "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T", "TGG": "W",
        "TAT": "Y", "TAC": "Y", "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
        "TAA": " ", "TGA": " ", "TAG": " "
    }
    
    BASES = ["A", "T", "C", "G"]
    
    @classmethod
    def get_templates(cls):
        """生成初始化的计数模板"""
        reversed_codon_table = defaultdict(list)
        for codon, aa in cls.CODON_TABLE.items():
            reversed_codon_table[aa].append(codon)
            
        return {
            "nu": {a: 0 for a in cls.BASES},
            "dinu": {a + b: 0 for a in cls.BASES for b in cls.BASES},
            "codon": {key: 0 for key in cls.CODON_TABLE.keys()},
            "aa": {aa: 0 for aa in reversed_codon_table.keys()}
        }

    @classmethod
    def get_feature_names(cls) -> List[str]:
        """生成特征列名列表，确保顺序一致"""
        templates = cls.get_templates()
        feature_groups = [
            ("nu", templates["nu"]),
            ("dinu", templates["dinu"]),
            ("dinu_bdg", templates["dinu"]),
            ("dinu_non_bdg", templates["dinu"]),
            ("codon_bias", templates["codon"]),
            ("aa_bias", templates["aa"])
        ]
        names = []
        for prefix, template in feature_groups:
            # 字典键的顺序在 Python 3.7+ 是插入顺序，这里为了保险起见，最好是确定的
            # 但由于模板初始化是确定的，这里直接用 keys() 即可
            names.extend([f"{prefix}|{k}" for k in template.keys()])
        return names

# 将核心计算逻辑提取为独立的函数，便于 pickle 序列化传给子进程
def _process_single_genome(genome_id: str, cds_list: List[str]) -> Tuple[str, np.ndarray, bool]:
    """
    处理单个基因组的所有 CDS 序列，计算特征向量。
    返回: (genome_id, feature_vector, is_valid)
    """
    templates = GenomicConstants.get_templates()
    
    # 深拷贝模板以避免污染
    feature_count = {
        "nu": deepcopy(templates["nu"]),
        "dinu": deepcopy(templates["dinu"]),
        "dinu_bdg": deepcopy(templates["dinu"]),
        "dinu_non_bdg": deepcopy(templates["dinu"]),
        "codon": deepcopy(templates["codon"]),
        "aa": deepcopy(templates["aa"])
    }
    
    valid_cds_found = False
    
    for cds_seq in cds_list:
        cds_seq = str(cds_seq).upper()
        seq_len = len(cds_seq)
        if seq_len == 0:
            continue
            
        # 截断非3倍数部分
        remainder = seq_len % 3
        if remainder != 0:
            cds_seq = cds_seq[:-remainder]
            seq_len = len(cds_seq)
            
        if seq_len == 0: 
            continue

        valid_cds_found = True
        
        # --- 计数逻辑 ---
        # 预先获取引用以减少查找开销
        fc_nu = feature_count["nu"]
        fc_codon = feature_count["codon"]
        fc_aa = feature_count["aa"]
        fc_dinu = feature_count["dinu"]
        fc_dinu_non_bdg = feature_count["dinu_non_bdg"]
        fc_dinu_bdg = feature_count["dinu_bdg"]
        codon_table = GenomicConstants.CODON_TABLE
        
        for i in range(0, seq_len, 3):
            codon = cds_seq[i:i + 3]
            
            # 1. Nucleotide
            for nu in codon:
                if nu in fc_nu: fc_nu[nu] += 1
            
            # 2. Codon & AA
            if codon in codon_table:
                fc_codon[codon] += 1
                fc_aa[codon_table[codon]] += 1
            
            # 3. Dinucleotide
            # Pos 1-2
            dinu_pos12 = cds_seq[i: i + 2]
            if dinu_pos12 in fc_dinu:
                fc_dinu[dinu_pos12] += 1
                fc_dinu_non_bdg[dinu_pos12] += 1
            
            # Pos 2-3
            dinu_pos23 = cds_seq[i + 1: i + 3]
            if dinu_pos23 in fc_dinu:
                fc_dinu[dinu_pos23] += 1
                fc_dinu_non_bdg[dinu_pos23] += 1
                
            # Pos 3-1 (Bridge)
            if i + 4 <= seq_len:
                dinu_bridge = cds_seq[i + 2: i + 4]
                if dinu_bridge in fc_dinu:
                    fc_dinu[dinu_bridge] += 1
                    fc_dinu_bdg[dinu_bridge] += 1

    if not valid_cds_found:
        logger.warning(f"Contig {genome_id} has no valid CDS sequences.")
        return genome_id, np.full(137, np.log2(0.0001)), False

    # --- 转换逻辑 (Transform) ---
    feature_dict = {}
    
    # 1. Nu Freq
    sum_nu = sum(feature_count["nu"].values())
    nu_freq = {k: v / sum_nu if sum_nu > 0 else 0 for k, v in feature_count["nu"].items()}
    feature_dict["nu"] = nu_freq
    
    # 2. Dinu Odds
    def calc_odds(count_dict, total_sum):
        res = {}
        for dinu, count in count_dict.items():
            expected = nu_freq.get(dinu[0], 0) * nu_freq.get(dinu[1], 0)
            if total_sum > 0 and expected > 0:
                res[dinu] = (count / total_sum) / expected
            else:
                res[dinu] = 0
        return res

    feature_dict["dinu"] = calc_odds(feature_count["dinu"], sum(feature_count["dinu"].values()))
    feature_dict["dinu_bdg"] = calc_odds(feature_count["dinu_bdg"], sum(feature_count["dinu_bdg"].values()))
    feature_dict["dinu_non_bdg"] = calc_odds(feature_count["dinu_non_bdg"], sum(feature_count["dinu_non_bdg"].values()))
    
    # 3. Codon Bias
    feature_dict["codon_bias"] = {}
    for codon, count in feature_count["codon"].items():
        aa = codon_table.get(codon)
        aa_count = feature_count["aa"].get(aa, 0)
        feature_dict["codon_bias"][codon] = count / aa_count if aa_count > 0 else 0
        
    # 4. AA Bias
    sum_aa = sum(feature_count["aa"].values())
    feature_dict["aa_bias"] = {k: v / sum_aa if sum_aa > 0 else 0 for k, v in feature_count["aa"].items()}
    
    # 5. Flatten & Log
    flat_vector = []
    # 顺序必须与 GenomicConstants.get_feature_names() 一致
    order_groups = [
        ("nu", templates["nu"]),
        ("dinu", templates["dinu"]),
        ("dinu_bdg", templates["dinu"]),
        ("dinu_non_bdg", templates["dinu"]),
        ("codon_bias", templates["codon"]),
        ("aa_bias", templates["aa"])
    ]
    
    for key, template in order_groups:
        sub_dict = feature_dict[key]
        for sub_key in template.keys():
            flat_vector.append(sub_dict.get(sub_key, 0))
            
    arr = np.array(flat_vector, dtype=np.float32)
    # Log2 transform, handling zeros
    arr[arr == 0] = 0.0001
    arr = np.log2(arr)
    
    return genome_id, arr, True


class GenomicTraitsEncoder:
    def __init__(self, genome_file: str, cds_file: str, n_jobs: int = -1):
        """
        初始化编码器。
        
        Args:
            genome_file: 基因组 FASTA 文件路径 (用于获取 ID 列表)
            cds_file: CDS FASTA 文件路径
            n_jobs: 并行进程数。-1 表示使用所有可用 CPU 核心。
        """
        self.genome_file = genome_file
        self.cds_file = cds_file
        self.n_jobs = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
        self.feature_names = GenomicConstants.get_feature_names()

    def _parse_and_group_cds(self) -> Dict[str, List[str]]:
        """
        解析 CDS 文件并按基因组 ID 分组。
        """
        logger.debug(f"     [DEBUG] Parsing CDS file: {self.cds_file}")
        
        # 获取目标基因组 ID 列表 (用于过滤或验证)
        # 注意：如果 genome_file 很大，这里只读取 ID 即可
        target_ids = set()
        try:
            for record in SeqIO.parse(self.genome_file, 'fasta'):
                target_ids.add(record.id)
        except Exception as e:
            logger.error(f"Error reading genome file: {e}")
            raise

        cds_dict = defaultdict(list)
        count = 0
        try:
            for record in SeqIO.parse(self.cds_file, 'fasta'):
                # 假设 ID 格式为 GenomeID_ProteinID
                # 使用 rsplit 确保只分割最后一个下划线
                genome_id = record.id.rsplit('_', 1)[0]
                
                # 只有当该 ID 在基因组文件中存在时才添加 (可选，取决于需求)
                if genome_id in target_ids:
                    cds_dict[genome_id].append(str(record.seq))
                    count += 1
                else:
                    # 也可以选择记录警告
                    pass
        except Exception as e:
            logger.error(f"Error reading CDS file: {e}")
            raise

        logger.debug(f"     [DEBUG] Loaded {count} CDS sequences for {len(cds_dict)} genomes.")
        
        # 确保所有 target_ids 都在字典中，即使没有 CDS (为了输出完整性)
        for gid in target_ids:
            if gid not in cds_dict:
                cds_dict[gid] = []
                
        return cds_dict

    def encode(self) -> pd.DataFrame:
        """
        执行多进程特征提取。
        
        Returns:
            pd.DataFrame: 索引为基因组 ID，列为 137 维特征。
        """
        cds_dict = self._parse_and_group_cds()
        
        results = []
        ids = []
        
        logger.debug(f"     [DEBUG] Starting genome trait extraction with {self.n_jobs} processes...")
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            # 提交任务
            future_to_id = {
                executor.submit(_process_single_genome, gid, seqs): gid 
                for gid, seqs in cds_dict.items()
            }
            
            # 收集结果
            for future in as_completed(future_to_id):
                gid = future_to_id[future]
                try:
                    genome_id, feature_vector, is_valid = future.result()
                    ids.append(genome_id)
                    results.append(feature_vector)
                    
                    if not is_valid:
                        logger.warning(f"Genome {gid} has no valid CDS sequences. Features set to 0.")
                        
                except Exception as e:
                    logger.error(f"Exception processing genome {gid}: {e}")
                    # 发生错误时填充 NaN，保持 DataFrame 形状
                    ids.append(gid)
                    results.append(np.full(137, np.nan))

        # 构建 DataFrame
        # 确保 ids 和 results 的顺序对应 (as_completed 是乱序的，但我们同步 append 了)
        df = pd.DataFrame(results, index=ids, columns=self.feature_names)
        
        # 按索引排序，使输出结果确定
        df.sort_index(inplace=True)

        logger.debug(f"     [DEBUG] Genome traite encoding complete. Shape: {df.shape}")
        return df

# class GenomicTraitsEncoder:
#     def __init__(self, genome_file, cds_file):
#         # --- 初始化基础字典 (只运行一次) ---
#         self.codon_table = {
#             "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
#             "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R", "AGA": "R", "AGG": "R",
#             "AAT": "N", "AAC": "N", "GAT": "D", "GAC": "D", "TGT": "C", "TGC": "C",
#             "CAA": "Q", "CAG": "Q", "GAA": "E", "GAG": "E", "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
#             "CAT": "H", "CAC": "H", "ATG": "M", "ATT": "I", "ATC": "I", "ATA": "I",
#             "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L", "TTA": "L", "TTG": "L",
#             "AAA": "K", "AAG": "K", "TTT": "F", "TTC": "F", "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
#             "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S", "AGT": "S", "AGC": "S",
#             "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T", "TGG": "W",
#             "TAT": "Y", "TAC": "Y", "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
#             "TAA": " ", "TGA": " ", "TAG": " "
#         }
#         self.genome_file = genome_file
#         self.cds_file = cds_file

#         genome_ids = [record.id for record in SeqIO.parse(genome_file, 'fasta')]
#         cds_dict = {genome_id:[] for genome_id in genome_ids}
#         for record in SeqIO.parse(cds_file, 'fasta'):
#             tmp_id = (record.id).rsplit('_', 1)[0]
#             if tmp_id not in cds_dict:
#                 raise ValueError()
#             cds_dict[tmp_id].append(str(record.seq))

#         self.cds_dict = cds_dict

#         self.reversed_codon_table = defaultdict(list)
#         for codon, aa in self.codon_table.items():
#             self.reversed_codon_table[aa].append(codon)

#         # 基础计数器模板
#         self.nu_template = {a: 0 for a in ["A", "T", "C", "G"]}
#         self.dinu_template = {a + b: 0 for a in ["A", "T", "C", "G"] for b in ["A", "T", "C", "G"]}
#         self.codon_template = {key: 0 for key in self.codon_table.keys()}
#         self.aa_template = {aa: 0 for aa in self.reversed_codon_table.keys()}

#         # 生成特征名称列表 (用于 DataFrame 列名)
#         self.feature_names = []
#         feature_groups = {
#             "nu": self.nu_template,
#             "dinu": self.dinu_template,
#             "dinu_bdg": self.dinu_template,
#             "dinu_non_bdg": self.dinu_template,
#             "codon_bias": self.codon_template,
#             "aa_bias": self.aa_template
#         }
#         for key in feature_groups:
#             self.feature_names += [key + "|" + name for name in list(feature_groups[key].keys())]
#         # 总共应该有 137 个特征

#     def _get_empty_counts(self):
#         """返回一个重置的计数器字典"""
#         return {
#             "nu": deepcopy(self.nu_template),
#             "dinu": deepcopy(self.dinu_template),
#             "dinu_bdg": deepcopy(self.dinu_template),
#             "dinu_non_bdg": deepcopy(self.dinu_template),
#             "codon": deepcopy(self.codon_template),
#             "aa": deepcopy(self.aa_template)
#         }

#     def _count_coding_seq(self, coding_seq, feature_count):
#         """统计单个 CDS 片段的特征"""
#         seq_len = len(coding_seq)
#         for i in range(0, seq_len, 3):
#             # 1. 确保不越界
#             if i + 3 > seq_len: break
            
#             codon = coding_seq[i:i + 3]
            
#             # 统计单核苷酸
#             for nu in codon:
#                 if nu in feature_count["nu"]:
#                     feature_count["nu"][nu] += 1
            
#             # 统计密码子和氨基酸
#             if codon in self.codon_table:
#                 feature_count["codon"][codon] += 1
#                 feature_count["aa"][self.codon_table[codon]] += 1
            
#             # 统计双核苷酸 (Dinucleotide)
#             # 位置 1-2 (Codon 内部)
#             dinu_pos12 = coding_seq[i: i + 2]
#             if dinu_pos12 in feature_count["dinu"]:
#                 feature_count["dinu"][dinu_pos12] += 1
#                 feature_count["dinu_non_bdg"][dinu_pos12] += 1
            
#             # 位置 2-3 (Codon 内部)
#             dinu_pos23 = coding_seq[i + 1: i + 3]
#             if dinu_pos23 in feature_count["dinu"]:
#                 feature_count["dinu"][dinu_pos23] += 1
#                 feature_count["dinu_non_bdg"][dinu_pos23] += 1
            
#             # 位置 3-1 (Bridge, 跨 Codon)
#             if i + 4 <= seq_len: # 确保下一个 codon 至少有一个碱基
#                 dinu_bridge = coding_seq[i + 2: i + 4]
#                 if dinu_bridge in feature_count["dinu"]:
#                     feature_count["dinu"][dinu_bridge] += 1
#                     feature_count["dinu_bdg"][dinu_bridge] += 1
                    
#         return feature_count

#     def _transform_feature(self, feature_count):
#         """将原始计数转换为频率/偏好性得分，并进行 Log 变换"""
#         feature_dict = {}

#         # 1. Nucleotide Frequency
#         sum_nu = sum(feature_count["nu"].values())
#         nu_freq = {nu: count / sum_nu if sum_nu > 0 else 0 for nu, count in feature_count["nu"].items()}
#         feature_dict["nu"] = nu_freq

#         # 2. Dinucleotide Odds Ratio (Observed / Expected)
#         # Formula: prob_xy / (prob_x * prob_y)
#         def calc_dinu_odds(count_dict, total_sum):
#             res = {}
#             for dinu, count in count_dict.items():
#                 denom = (nu_freq[dinu[0]] * nu_freq[dinu[1]])
#                 if total_sum > 0 and denom > 0:
#                     res[dinu] = (count / total_sum) / denom
#                 else:
#                     res[dinu] = 0
#             return res

#         sum_dinu = sum(feature_count["dinu"].values())
#         feature_dict["dinu"] = calc_dinu_odds(feature_count["dinu"], sum_dinu)

#         sum_dinu_bdg = sum(feature_count["dinu_bdg"].values())
#         feature_dict["dinu_bdg"] = calc_dinu_odds(feature_count["dinu_bdg"], sum_dinu_bdg)

#         sum_dinu_non_bdg = sum(feature_count["dinu_non_bdg"].values())
#         feature_dict["dinu_non_bdg"] = calc_dinu_odds(feature_count["dinu_non_bdg"], sum_dinu_non_bdg)

#         # 3. Codon Bias (RSCU-like)
#         # Formula: count_codon / count_aa
#         feature_dict["codon_bias"] = {}
#         for codon, count in feature_count["codon"].items():
#             aa = self.codon_table[codon]
#             aa_count = feature_count["aa"][aa]
#             if aa_count > 0:
#                 feature_dict["codon_bias"][codon] = count / aa_count
#             else:
#                 feature_dict["codon_bias"][codon] = 0

#         # 4. Amino Acid Frequency
#         sum_aa = sum(feature_count["aa"].values())
#         feature_dict["aa_bias"] = {aa: count / sum_aa if sum_aa > 0 else 0 
#                                    for aa, count in feature_count["aa"].items()}

#         # 5. Flatten and Log Transform
#         # 顺序必须与 self.feature_names 保持一致
#         feature_array = []
#         # 注意：这里遍历顺序要和 feature_names 生成时的顺序严格一致
#         # feature_names 生成顺序: nu, dinu, dinu_bdg, dinu_non_bdg, codon_bias, aa_bias
#         order_keys = ["nu", "dinu", "dinu_bdg", "dinu_non_bdg", "codon_bias", "aa_bias"]
        
#         for key in order_keys:
#             # 确保内部字典的顺序也是固定的 (Python 3.7+ 字典有序，但为了保险起见，最好按 key 排序或使用模板顺序)
#             sub_dict = feature_dict[key]
#             # 使用模板的 key 顺序来提取值，确保对齐
#             if key == "nu": template_keys = list(self.nu_template.keys())
#             elif "dinu" in key: template_keys = list(self.dinu_template.keys())
#             elif key == "codon_bias": template_keys = list(self.codon_template.keys())
#             elif key == "aa_bias": template_keys = list(self.aa_template.keys())
            
#             for sub_key in template_keys:
#                 feature_array.append(sub_dict.get(sub_key, 0))

#         feature_array = np.array(feature_array)
#         # Log2 transform (handling zeros)
#         feature_array[feature_array == 0] = 0.0001
#         feature_array = np.log2(feature_array)
        
#         return feature_array

#     def encode(self):
#         """
#         主入口函数
#         :param cds_dict: 字典，格式为 { 'Sequence_ID': ['CDS_SEQ_1', 'CDS_SEQ_2', ...] }
#                          注意：这里的 CDS_SEQ 应该是核苷酸序列字符串
#         :return: (DataFrame, list) -> (特征矩阵, 无蛋白序列ID列表)
#         """
#         matrix = []
#         ids = []
#         no_protein_ids = []
#         cds_dict = self.cds_dict

#         for seq_id, cds_list in cds_dict.items():
#             ids.append(seq_id)
            
#             # 初始化计数器
#             feature_count = self._get_empty_counts()
#             valid_cds_found = False

#             if cds_list and len(cds_list) > 0:
#                 for cds_seq in cds_list:
#                     cds_seq = str(cds_seq).upper()
#                     # 简单过滤：长度必须是3的倍数且大于0
#                     if len(cds_seq) > 0: 
#                         # 如果不是3的倍数，通常 prodigal 会输出完整的，但为了安全截断
#                         remainder = len(cds_seq) % 3
#                         if remainder != 0:
#                             cds_seq = cds_seq[:-remainder]
                        
#                         self._count_coding_seq(cds_seq, feature_count)
#                         valid_cds_found = True
            
#             if valid_cds_found:
#                 feature_vector = self._transform_feature(feature_count)
#             else:
#                 feature_vector = np.full([137], np.nan)
#                 no_protein_ids.append(seq_id)
            
#             matrix.append(feature_vector)

#         # 转换为 DataFrame
#         df_features = pd.DataFrame(np.array(matrix), index=ids, columns=self.feature_names)
#         return df_features
