# ADR-0001 — Máquina de estado explicita

**Status:** aceito
**Data:** 2026-07-31

## Contexto
A máquina de estados foi escolhida em vez do React porque lidamos com processos irreversíveis que exigem controle rigoroso do fluxo de execução. Dessa forma, não deixamos que o modelo decida qual será o próximo passo: todas as transições do grafo são previamente definidas.

Em um fluxo de pagamento, por exemplo, a sequência de etapas não é uma decisão a ser tomada pelo modelo. Não é possível liquidar uma operação antes de realizar a cotação. A máquina de estados garante que essa ordem seja sempre respeitada.

Com essa abordagem, temos as seguintes premissas:
- Transições ilegais são tratadas como erros de implementação, e não como comportamentos inesperados do modelo.
- Cada nó possui contratos de entrada e saída fortemente tipados.
- O fluxo é determinístico e totalmente auditável.

## Decisão
- Máquina de estado explicita
Determinismo é no nível do grafo de compra (plan → retrieve → quote → mandate → confirm → settle); dentro do nó retrieve, a pesquisa é deliberadamente agêntica. O que nunca é agêntico: autorização, valor, e transição entre os 6 estados.

## Alternativas consideradas
- ReAct livre (agente decide toda ação a cada passo) - descartado porque a garantia de ordem ficaria dentro do raciocínio do LLM.
- Máquina de estados também dentro do nó retrieve — descartado porque pesquisa é tarefa aberta legítima; reduzir a liberdade ali sem necessidade não traz ganho de segurança e piora a qualidade do retrieval.

## Consequências
- Positiva: transição ilegal é erro de programa (exceção antes de qualquer efeito colateral), não comportamento indesejado a ser corrigido depois; trace mostra os estados percorridos, o que faz false_settlement_rate ser uma métrica comsignificado real.
- Custo aceito: o sistema é menos flexível a novos tipos de compra sem alterar o grafo; adicionar um estado exige mudança de código, não só de prompt.
- O que precisaria mudar para revisitar: se o domínio deixasse de ter regras de ordem rígidas (por exemplo, um assistente só consultivo, sem liquidação real), o custo do grafo fixo deixaria de se justificar.
