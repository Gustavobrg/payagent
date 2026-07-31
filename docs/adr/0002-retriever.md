# ADR-0002 — Retriever determinístico, decisão de busca fica no sub-agente

**Status:** aceito
**Data:** 2026-07-31

## Contexto
O RAG precisa responder com citação do chunk recuperado (invariante I4), mas o próprio conteúdo recuperado é o principal vetor de prompt injection do sistema. Busca por similaridade vetorial pura, devolve ruído demais para grounding sério. E decidir o que procurar (catálogo, política, os dois, reformular ou não) é tarefa aberta demais para travar numa sequência fixa sem perder qualidade de retrieval.

## Decisão
- Retrieval em duas camadas. `Retriever` é o mecanismo: embed da query, busca vetorial no Qdrant, rerank por cross-encoder, filtro exato/faixa validado contra um conjunto de chaves conhecido por coleção (chave fora disso levanta `ValueError`, não ignora). O nó `retrieve` é quem decide: um sub-agente com tool calling, orçamento fixo de iterações, que escolhe sozinho se chama `search_catalog`, `search_policies`, `get_sku_details` ou `compare_skus`, e se reformula a query.
- Todo `ToolChunk` devolvido pelas tools (`rag/tools.py`) já sai delimitado com uma tag marcando o conteúdo como não confiável (invariante I5). Isso é aplicado na borda das tools, não dentro do `Retriever` nem do grafo. Nenhum consumidor recebe texto recuperado sem a marcação.
- `Embedder` e `Reranker` são interfaces injetáveis: implementação real via OpenRouter, implementação determinística (hash + overlap de palavras, sem rede) para teste. O sub-agente e o `Retriever` inteiro são exercitáveis em CI sem chave de API.

## Alternativas consideradas
- Busca vetorial pura, sem rerank: descartamos porque top-k por similaridade de embedding sozinho é ruidoso demais para citação confiável. O cross-encoder corrige a ordenação a baixo custo.
- Ignorar silenciosamente um filtro por chave desconhecida também ficou de fora: um typo, ou um filtro de coleção errada, ampliaria o escopo da busca sem avisar ninguém. Preferimos erro explícito, que o sub-agente vê e pode corrigir reformulando.

## Consequências
- Positiva: o pipeline de busca é testável isoladamente do LLM; o sub-agente pode ser reprompt-ado, trocado de modelo ou mockado sem tocar no `Retriever`.
- Custo aceito: duas camadas de abstração (`Retriever` + sub-agente) para algo que poderia ser uma função só; mais um contrato de filtro por coleção para manter sincronizado se o schema do catálogo mudar.
- O que precisaria mudar para revisitar: se o volume de catálogo justificasse busca híbrida (BM25 + vetor) ou um reranker bem mais caro, o pipeline de duas camadas teria que absorver um terceiro estágio sem quebrar o contrato de `RetrievedChunk`.
