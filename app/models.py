from datetime import datetime
from sqlalchemy import func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

class Base(DeclarativeBase):          # a "raiz" de todos os models; o Alembic lê o Base.metadata
    pass

class PolicyDocument(Base):
    __tablename__ = "policy_document"

    id: Mapped[int] = mapped_column(primary_key=True)        # PK auto-incremento
    seguradora: Mapped[str]                                  # NOT NULL por padrão
    produto: Mapped[str]
    susep_processo: Mapped[str]
    versao: Mapped[str | None]                               # nullable (o | None)
    tipo_imovel: Mapped[str | None]
    pdf_url: Mapped[str]
    pdf_hash: Mapped[str]
    extracted_at: Mapped[datetime] = mapped_column(server_default=func.now())  # default no banco
