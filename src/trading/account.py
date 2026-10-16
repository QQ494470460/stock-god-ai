# -*- coding: utf-8 -*-
"""
账号管理模块
加密存储券商登录凭证，支持多账号管理
"""

import json
import base64
import hashlib
import secrets
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict, field
from loguru import logger

from config.settings import CONFIG_DIR

# 账号文件落在 CONFIG_DIR（源码=项目 config；打包=exe 同级 config，重建不丢账号）
ACCOUNTS_FILE = CONFIG_DIR / "broker_accounts.enc"


@dataclass
class BrokerAccount:
    """券商账号"""
    broker: str                # 券商名称 (yuekai / guojin / ...)
    account_id: str            # 资金账号
    name: str = ""             # 账号别名
    qmt_path: str = ""         # QMT安装目录路径
    qmt_account: str = ""      # QMT登录账号
    qmt_password_enc: str = "" # 加密后的密码
    created_at: str = ""
    last_login: str = ""
    active: bool = False
    
    def to_dict(self) -> dict:
        d = asdict(self)
        # 不序列化敏感字段
        return {k: v for k, v in d.items() if not k.endswith("_enc")}


class AccountManager:
    """券商账号管理器 — 加密存储"""
    
    _KEY = "stock-god-ai-qclaw-broker-key-v1"  # 固定密钥混入随机盐
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = Path(config_dir) if config_dir else CONFIG_DIR
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._accounts: Dict[str, BrokerAccount] = {}
        # ★ 账号文件必须跟随 self.config_dir（此前误用模块级全局路径，
        #   导致传入 config_dir 时不生效，甚至写到真实账号文件）
        self._accounts_file = self.config_dir / "broker_accounts.enc"
        self._salt_file = self.config_dir / ".salt"
        self._load()
    
    @property
    def accounts(self) -> List[BrokerAccount]:
        return list(self._accounts.values())
    
    def get(self, account_id: str) -> Optional[BrokerAccount]:
        return self._accounts.get(account_id)
    
    def get_active(self) -> Optional[BrokerAccount]:
        """获取当前激活的账号"""
        for a in self._accounts.values():
            if a.active:
                return a
        return None
    
    def get_decrypted_password(self, account_id: str) -> Optional[str]:
        """获取解密后的密码（解密失败返回 None，绝不抛异常拖垮调用方）"""
        acct = self._accounts.get(account_id)
        if not acct or not acct.qmt_password_enc:
            return None
        try:
            return self._decrypt(acct.qmt_password_enc)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"账号 {account_id} 密码解密失败（可能盐值变更/数据损坏）: {e}")
            return None
    
    def add_or_update(
        self,
        broker: str,
        account_id: str,
        name: str = "",
        qmt_path: str = "",
        qmt_account: str = "",
        qmt_password: str = "",
        set_active: bool = True,
    ) -> BrokerAccount:
        """添加或更新账号

        更新语义：显式传入的字段才覆盖，未传的保留原值
        （尤其是密码——只更新 QMT 路径时不应把已存密码清空）。
        """
        is_new = account_id not in self._accounts
        prev = self._accounts.get(account_id)

        # 未显式给定的字段回退到已有值（首次创建则为空）
        eff_name = name or (prev.name if prev else f"{broker}_{account_id[-4:]}")
        eff_qmt_path = qmt_path or (prev.qmt_path if prev else "")
        eff_qmt_account = qmt_account or (prev.qmt_account if prev else "")
        if qmt_password:
            eff_pwd_enc = self._encrypt(qmt_password)
        else:
            eff_pwd_enc = prev.qmt_password_enc if prev else ""

        acct = BrokerAccount(
            broker=broker,
            account_id=account_id,
            name=eff_name,
            qmt_path=eff_qmt_path,
            qmt_account=eff_qmt_account,
            qmt_password_enc=eff_pwd_enc,
            created_at=prev.created_at if prev else datetime.now().isoformat(),
            last_login=datetime.now().isoformat(),
            active=set_active,
        )
        
        if set_active:
            for a in self._accounts.values():
                a.active = False
        
        self._accounts[account_id] = acct
        self._save()
        logger.info(f"账号 {'新增' if is_new else '更新'}: {account_id} ({broker})")
        return acct
    
    def remove(self, account_id: str) -> bool:
        if account_id in self._accounts:
            del self._accounts[account_id]
            self._save()
            logger.info(f"账号已删除: {account_id}")
            return True
        return False
    
    def set_active(self, account_id: str) -> bool:
        if account_id in self._accounts:
            for a in self._accounts.values():
                a.active = False
            self._accounts[account_id].active = True
            self._save()
            return True
        return False
    
    # ========== 加密/解密 ==========
    
    def _get_salt(self) -> str:
        """获取或生成盐"""
        if not self._salt_file.exists():
            salt = secrets.token_hex(16)
            self._salt_file.write_text(salt, encoding="utf-8")
            return salt
        return self._salt_file.read_text(encoding="utf-8").strip()
    
    def _derive_key(self) -> str:
        """从固定密钥+随机盐派生AES密钥"""
        salt = self._get_salt()
        return hashlib.pbkdf2_hmac(
            "sha256",
            self._KEY.encode(),
            salt.encode(),
            100_000,
            dklen=32
        ).hex()
    
    def _encrypt(self, plaintext: str) -> str:
        """简易加密（混入派生密钥的XOR + base64）"""
        if not plaintext:
            return ""
        key = self._derive_key()
        key_bytes = bytes.fromhex(key)
        plain_bytes = plaintext.encode("utf-8")
        
        # XOR with cycling key
        encrypted = bytes(
            p ^ key_bytes[i % len(key_bytes)]
            for i, p in enumerate(plain_bytes)
        )
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def _decrypt(self, ciphertext: str) -> str:
        """解密"""
        if not ciphertext:
            return ""
        key = self._derive_key()
        key_bytes = bytes.fromhex(key)
        encrypted = base64.urlsafe_b64decode(ciphertext.encode())
        
        plain_bytes = bytes(
            e ^ key_bytes[i % len(key_bytes)]
            for i, e in enumerate(encrypted)
        )
        return plain_bytes.decode("utf-8")
    
    def _save(self):
        """持久化（不含密码明文）"""
        data = {
            account_id: {
                "broker": a.broker,
                "account_id": a.account_id,
                "name": a.name,
                "qmt_path": a.qmt_path,
                "qmt_account": a.qmt_account,
                "qmt_password_enc": a.qmt_password_enc,
                "created_at": a.created_at,
                "last_login": a.last_login,
                "active": a.active,
            }
            for account_id, a in self._accounts.items()
        }
        with open(self._accounts_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.debug(f"账号数据已保存: {len(data)} 个账号")
    
    def _load(self):
        """加载持久化数据"""
        self._accounts.clear()
        if not self._accounts_file.exists():
            return
        
        try:
            with open(self._accounts_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for aid, d in data.items():
                self._accounts[aid] = BrokerAccount(
                    broker=d.get("broker", ""),
                    account_id=d.get("account_id", aid),
                    name=d.get("name", ""),
                    qmt_path=d.get("qmt_path", ""),
                    qmt_account=d.get("qmt_account", ""),
                    qmt_password_enc=d.get("qmt_password_enc", ""),
                    created_at=d.get("created_at", ""),
                    last_login=d.get("last_login", ""),
                    active=d.get("active", False),
                )
            logger.info(f"已加载 {len(self._accounts)} 个券商账号")
        except Exception as e:
            logger.error(f"账号文件加载失败: {e}")


# 全局单例
account_manager = AccountManager()
