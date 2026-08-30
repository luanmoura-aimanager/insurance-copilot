# Política de Privacidade — Insurance Copilot

Última atualização: 30 de agosto de 2026.

## O que este sistema é

O Insurance Copilot é um **projeto pessoal de portfólio**, não um serviço comercial. Ele
responde perguntas sobre a **estrutura de coberturas** de condições gerais de seguro
residencial registradas na SUSEP — documentos públicos, que descrevem produtos, não
apólices de clientes.

As respostas são geradas automaticamente por modelos de linguagem e **não são
aconselhamento de seguros**. Elas não substituem a leitura da apólice contratada nem a
orientação de um corretor ou da seguradora.

## Que dados são recebidos

O único canal de entrada de dados pessoais é o WhatsApp. Quando você envia uma mensagem
de texto para o número do projeto, são recebidos e armazenados:

- **o seu número de telefone** (o remetente informado pela Meta);
- **o conteúdo da mensagem de texto** que você enviou;
- o **identificador da mensagem** atribuído pela Meta e o identificador do número do
  projeto que recebeu a mensagem;
- os horários de recebimento e de conclusão do atendimento, um contador de tentativas e
  a **resposta gerada** para aquela mensagem.

Somente mensagens **de texto** são armazenadas. Áudio, imagem, documento e outros tipos
de mensagem não geram registro no banco de dados.

Não há coleta a partir de nenhuma outra fonte: não existe cadastro, formulário, cookie,
rastreamento de navegação, nem qualquer enriquecimento de dados com bases de terceiros.
Nada além do que você escreve no WhatsApp é coletado sobre você.

## Para que os dados são usados

Exclusivamente para **gerar e entregar a resposta à pergunta que você enviou**. Não há
uso comercial, não há publicidade, não há perfilamento, e os dados não são vendidos nem
cedidos a ninguém.

Para produzir a resposta, o conteúdo da sua pergunta trafega pelos provedores de
infraestrutura estritamente necessários:

- **Meta (WhatsApp Cloud API)** — entrega a sua mensagem ao sistema e leva a resposta de
  volta até você;
- **Anthropic** — os modelos de linguagem que interpretam a pergunta e redigem a
  resposta recebem o texto da sua mensagem;
- **Voyage AI** — quando a pergunta exige busca no texto das cláusulas, o texto da
  pergunta é enviado para gerar a representação vetorial usada na busca;
- **Railway** — hospeda a aplicação e o banco de dados onde a mensagem e a resposta
  ficam gravadas.

O seu número de telefone é enviado apenas à Meta, para entregar a resposta. Ele não é
enviado à Anthropic nem à Voyage AI.

## Registros de operação (logs)

Os logs da aplicação são deliberadamente pobres em dado pessoal:

- o **conteúdo das mensagens nunca é registrado** em log, em nenhum nível — apenas o
  tamanho do texto;
- o **número de telefone é mascarado**, restando apenas os quatro últimos dígitos;
- o identificador da mensagem atribuído pela Meta é registrado apenas como um resumo
  criptográfico curto, porque o identificador original embute o número de telefone;
- mensagens de erro de bibliotecas não são registradas na íntegra, justamente porque
  poderiam carregar consigo os valores enviados ao banco de dados.

## Retenção e exclusão

**Não existe expurgo automático.** Hoje o sistema não implementa nenhuma rotina que apague
mensagens após um prazo: a linha correspondente à sua mensagem — número, texto, resposta
e horários — permanece no banco de dados por tempo indeterminado, mesmo depois de a
resposta ter sido enviada ou de o atendimento ter falhado. Definir uma janela de retenção
é uma pendência conhecida e registrada no repositório; enquanto ela não existir, este
documento não promete prazo nenhum.

**A exclusão é feita sob pedido.** Basta escrever para o endereço abaixo informando o
número de telefone usado, que os registros correspondentes serão apagados do banco de
dados. O mesmo endereço atende pedidos de acesso aos dados guardados sobre você.

Registros mantidos pelos provedores acima (por exemplo, o histórico da conversa no seu
próprio aplicativo do WhatsApp, ou logs internos da Meta) seguem as políticas deles e
estão fora do alcance deste projeto.

## Contato

luanmisaelmoura@gmail.com
