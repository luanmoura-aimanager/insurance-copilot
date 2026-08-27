"""whatsapp_message: duas fases (answer + attempts + status 'computed')

Revision ID: e7b214d90c35
Revises: c5a71b3e9d84
Create Date: 2026-08-27 09:14:02.331907

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e7b214d90c35'
down_revision: Union[str, Sequence[str], None] = 'c5a71b3e9d84'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Abre espaço pras duas fases da resposta: computar, depois enviar.

    Sem REVOKE novo, ao contrário da W2a: privilégio é de TABELA, e `whatsapp_message` já
    saiu do alcance do `insurance_ro` lá. Coluna nova numa tabela revogada nasce revogada.
    """
    # 1. A resposta, que passa a existir ANTES de ser enviada. É esta coluna que torna a
    #    retentativa de envio barata: sem ela, um envio que falha depois do grafo obrigaria
    #    a re-perguntar, pagando outra rodada de LLM pela mesma pergunta.
    #
    #    `sa.Text()` aqui exige `mapped_column(Text)` no modelo: `alembic check` é enforced
    #    e `compare_type` é o default, então TEXT de um lado e VARCHAR do outro viraria um
    #    diff perpétuo numa migration sobre outro assunto.
    op.add_column('whatsapp_message', sa.Column('answer', sa.Text(), nullable=True))

    # 2. O orçamento de tentativas. `server_default='0'` não é só pra viabilizar o
    #    `ADD COLUMN NOT NULL` numa tabela com linhas: o INSERT do webhook é Core e não
    #    lista a coluna, então a autoridade do valor inicial é o banco — mesmo argumento
    #    que já está escrito no `status`.
    #
    #    Não há coluna de ERRO, e isso é decisão: o texto de uma exceção do SQLAlchemy
    #    carrega `[parameters: (...)]`, que aqui é o telefone e a mensagem inteira do
    #    usuário (achado da W2a). Erro vai pro log, montado à mão; a linha guarda a contagem.
    op.add_column(
        'whatsapp_message',
        sa.Column('attempts', sa.Integer(), nullable=False, server_default='0'),
    )

    # 3. O quarto estado. `computed` = "a resposta existe no banco e ainda não saiu" — é o
    #    estado intermediário que separa o que foi pago do que foi entregue.
    op.drop_constraint('ck_whatsapp_message_status', 'whatsapp_message', type_='check')
    op.create_check_constraint(
        'ck_whatsapp_message_status',
        'whatsapp_message',
        "status IN ('pending', 'computed', 'answered', 'failed')",
    )


def downgrade() -> None:
    """Volta pros três estados — e descarta respostas JÁ PAGAS.

    O `UPDATE` tem que vir ANTES do check antigo: com uma linha em `computed`, o
    `CREATE CONSTRAINT` falharia ao validar a tabela e o downgrade morreria no meio.

    O que se perde aqui é diferente do que a W2a perdia, e o contraste é a lição. Lá o
    downgrade descartava PERGUNTAS, que não se reconstroem de lugar nenhum (a Meta não
    reenvia o que já recebeu 200). Aqui descarta RESPOSTAS, que se reconstroem — a próxima
    varredura repergunta ao grafo. Ou seja: o dano é dinheiro, não informação. Volta pra
    `pending` de propósito, e não pra `failed`: a pergunta continua legítima e sem resposta.
    """
    op.execute("UPDATE whatsapp_message SET status = 'pending' WHERE status = 'computed'")

    op.drop_constraint('ck_whatsapp_message_status', 'whatsapp_message', type_='check')
    op.create_check_constraint(
        'ck_whatsapp_message_status',
        'whatsapp_message',
        "status IN ('pending', 'answered', 'failed')",
    )

    op.drop_column('whatsapp_message', 'attempts')
    op.drop_column('whatsapp_message', 'answer')
