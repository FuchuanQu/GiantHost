import multiprocessing
import os
import numpy as np
import pandas as pd
from src.core.base_feature import BaseFeatureExtractor
import pyhmmer
import concurrent.futures
import subprocess
from src.utils.logger import get_logger
logger = get_logger(__name__)

class GVOGOneHotExtractor(BaseFeatureExtractor):
    def __init__(self, config):
        """
        config 字典期望包含:
            gvog_list_path: str, GVOG 列表文件路径 (每行一个 GVOG ID)
        """
        super().__init__(config)
        self.evalue = 1e-5
        self.threads = os.cpu_count()
    def load(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Feature file not found at: {file_path}")
        return pd.read_parquet(file_path)

    def generate(self, fasta_path, output_path):
        # 确保存在预测的蛋白质文件
        cds_path = fasta_path.rsplit('.', 1)[0] + f'_{self.config["paths"]["source_suffix"]["cds"]}'
        prot_path = fasta_path.rsplit('.', 1)[0] + f'_{self.config["paths"]["source_suffix"]["protein"]}'
        if not os.path.exists(cds_path) or not os.path.exists(prot_path):
            logger.info(f"      CDS or PROTEIN file not found. Run gene prediction first.")
            command = [
                'python', self.prodigal_path,
                '-t', str(self.threads),
                '-q',
                '-i', fasta_path,
                '-d', cds_path,
                '-a', prot_path,
                # '-p', 'meta'
            ]
            subprocess.run(command,stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        # 加载 Contig 列表
        # logger.info(f"     [DEBUG] 加载 Contig 列表和蛋白质序列...")
        amino_alphabet = pyhmmer.easel.Alphabet.amino()
        dna_alphabet = pyhmmer.easel.Alphabet.dna()
        all_contigs = []
        with pyhmmer.easel.SequenceFile(fasta_path, digital=True, alphabet=dna_alphabet) as fna_file:
            for seq in fna_file:
                all_contigs.append(seq.name)

        # logger.info(f"     [DEBUG] 共识别到 {len(all_contigs)} 个 Contigs。")
        # 必须加载为 DigitalSequenceBlock 以便在多线程间共享内存
        with pyhmmer.easel.SequenceFile(prot_path, digital=True, alphabet=amino_alphabet) as faa_file:
            # 读取所有序列到内存
            seq_list = list(faa_file)
            sequences_block = pyhmmer.easel.DigitalSequenceBlock(amino_alphabet, seq_list)
        logger.debug(f"     [DEBUG] 已加载 {len(sequences_block)} 条蛋白质序列到内存块。")

        # 加载 HMM 模型
        HMM_FILE = self.config['paths']['gvog_hmm']
        with pyhmmer.plan7.HMMFile(HMM_FILE) as hmm_file:
            hmms = list(hmm_file)
        all_hmm_names = [hmm.name for hmm in hmms]
        logger.debug(f"     [DEBUG] 已加载 {len(hmms)} 个 HMM 模型。")

        valid_hits_collection = set()

        # 并行搜索
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.threads) as executor:
            # 提交任务
            future_to_hmm = {
                executor.submit(search_single_hmm, hmm, sequences_block): hmm 
                for hmm in hmms
            }

            # 处理结果
            for future in concurrent.futures.as_completed(future_to_hmm):
                hmm = future_to_hmm[future]
                hmm_name = hmm.name
                
                try:
                    hits = future.result()
                    for hit in hits:
                        if hit.evalue <= self.evalue:
                            protein_name = hit.name
                            # 解析 Contig ID
                            contig_name = get_contig_id_from_protein_id(protein_name)
                            
                            # 记录这个 Contig 含有这个 Gene
                            valid_hits_collection.add((contig_name, hmm_name))
                            
                except Exception as exc:
                    logger.error(f"模型 {hmm_name} 搜索时发生异常: {exc}")

        logger.debug(f"     [DEBUG] 搜索完成。共找到 {len(valid_hits_collection)} 个符合阈值的 Contig-Gene 关联。")

        df = pd.DataFrame(0, index=all_contigs, columns=all_hmm_names, dtype=int)

        # 填充矩阵
        # 这种迭代填充在 Pandas 中不是最快，但逻辑最清晰。
        # 对于中等规模数据（几万行x几百列）是完全没问题的。
        for contig, gene in valid_hits_collection:
            if contig in df.index:
                df.loc[contig, gene] = 1
            else:
                # 这种情况通常发生在你过滤掉了某些 Contig 或者 ID 解析函数写错了
                logger.warning(f"警告: 蛋白质 {contig} 对应的 Contig 不在原始 fna 文件中")
                pass

        # 保存结果
        df.to_parquet(output_path)
        logger.debug(f"     [DEBUG] GVOG One-Hot 特征矩阵已保存到: {output_path}")           

def get_contig_id_from_protein_id(protein_id_str):
    """
    关键函数：定义如何从蛋白质ID解析出Contig ID。
    
    假设1: 蛋白质ID就是Contig ID (如用户描述: contig_0)
    return protein_id_str
    
    假设2: 常见格式 ContigID_GeneNum (如 k141_1_1 -> k141_1)
    return "_".join(protein_id_str.split("_")[:-1])
    """
    # 这里根据你的描述，假设蛋白质ID直接对应Contig，或者你需要根据实际情况修改这里
    # 如果你的蛋白质ID是 "contig_0_1" 代表 contig_0 的第一个基因，请取消下一行的注释：
    return protein_id_str.rsplit("_", 1)[0]
    
def search_single_hmm(hmm, sequences):
    """
    单个 HMM 的搜索任务，将在线程池中运行。
    """
    pipeline = pyhmmer.plan7.Pipeline(alphabet=hmm.alphabet)
    hits = pipeline.search_hmm(hmm, sequences)
    return hits
