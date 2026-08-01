# ADR-0004 — Autorização determinística fora do guardrail

**Status:** aceito
**Data:** 2026-08-01

## Contexto

Guardrails (NeMo, Llama Guard) são classificadores probabilísticos: avaliam texto e
retornam uma decisão com alguma margem de erro. Autorização de pagamento — limite de
valor, allowlist de merchant, categoria restrita, exigência de step-up — não pode
herdar essa margem. Uma classificação errada em 1 a cada 1000 casos é aceitável para
bloquear conteúdo ofensivo; é inaceitável para autorizar uma liquidação. Se essa lógica
vivesse em Colang, no prompt de sistema, ou em qualquer decisão do LLM, a garantia de
segurança do sistema ficaria do tamanho da taxa de acerto do modelo.

## Decisão

Toda decisão de autorização é código Python determinístico em `src/payagent/policy/`,
sem chamada de LLM e sem import de `rag/`, `mcp_server/` ou qualquer biblioteca de
modelo (garantido por teste ast). `PolicyDecision` é uma união selada `Allow | Deny`:
`Allow` não é construível sem um `EffectGrant` concreto, e `evaluate()` — o único ponto
de entrada permitido, nunca `engine.decide()` diretamente — converte qualquer exceção,
tipo de retorno inesperado ou ação não mapeada em `Deny`. As regras reais
(`RulesPolicyEngine`: limite por transação, agregado 24h, allowlist de merchant,
categoria restrita, step-up) vêm de configuração injetada — nunca lidas de dentro de
`policy/` — e é isso que mantém `decide()` uma função pura de `(action, context)`.
`DenyAllPolicyEngine` continua fail-closed por padrão: é o que `mcp_server/__main__.py`
usa quando a configuração de política está ausente ou malformada, então a falha de
configuração nunca vira permissão silenciosa. Cada tool com efeito colateral recebe o
`EffectGrant` como argumento posicional obrigatório, tornando o efeito inalcançável por
qualquer caminho que não tenha passado por um `Allow`.

Confirmação humana (step-up) segue a mesma lógica: `PolicyContext.step_up_satisfied`
nunca é lido de um argumento de tool — vem de um `StepUpVerifier` injetado, que consulta
um desafio WebAuthn resolvido por canal separado da conversa. Nenhum modelo de request
tem campo com forma de autorização (`step_up_token`, `user_confirmed` ou similar).

## Alternativas consideradas

- **Regras de negócio em Colang (NeMo dialog rails).** Descartada porque Colang é
  linguagem de política de conteúdo, avaliada por classificador; misturar autorização
  financeira ali reintroduz probabilidade onde precisa haver certeza.
- **`allowed: bool` simples em vez de união selada.** Descartada porque um campo
  booleano é um typo (`if decision.allowed is not False`) de distância do fail-open;
  a união `Allow | Deny` torna esse erro impossível de compilar.
- **Aceitar `step_up_token` como argumento de tool, validado depois pelo policy
  engine.** Descartada porque o campo em si já é a superfície de ataque — é exatamente
  a forma que a categoria P5 ("já autorizei antes") teria para funcionar, independente
  de quão rigorosa fosse a validação por trás.