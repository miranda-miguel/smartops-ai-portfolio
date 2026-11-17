# 🚀 SmartOps AI — Agente de Triagem Logística, Reembolsos & Retenção

Protótipo funcional de **AI Operations** para triagem de chamados de suporte, análise de extravios logísticos e decisão de reembolso em operações de **Food Delivery**.

**Demo ao vivo:** COLE-AQUI-O-LINK-DO-STREAMLIT

---

## 🎯 O Problema de Negócio
Em operações de delivery de alta escala, o suporte N1 enfrenta três atritos principais:
1. **Alto tempo de triagem:** analisar relatos em linguagem natural manualmente gera filas em horários de pico.
2. **Risco de churn:** clientes legítimos que sofrem extravio precisam de resolução rápida.
3. **Abuso de reembolso:** aprovações automáticas sem cruzar o histórico recente geram perdas financeiras.

---

## 💡 A Solução (LLM + Guardrails Determinísticos)
O **SmartOps AI** combina **Structured Outputs (Pydantic + Gemini)** com uma **camada determinística de guardrails em Python**, em uma interface **Streamlit**:

* **Classificação e alocação de responsabilidade:** identifica a falha (*Extravio, Atraso, Item Faltante, Violação de Embalagem ou Suspeita de Fraude*) e aponta o elo responsável (*Malha Logística, Parceiro ou Plataforma*).
* **Guardrails em Python (`aplicar_guardrails`):** regras executadas após a inferência do modelo que impedem reembolso maior que o valor do pedido, corrigem valores negativos e forçam o bloqueio preventivo para auditoria humana quando a conta tem 4 ou mais reembolsos nos últimos 30 dias em pedidos acima de R$ 80.
* **Payload para n8n:** estrutura a decisão final em um JSON padronizado para consumo por workflows externos.

---

## 🛠️ Stack
* **Linguagem e validação:** Python 3.11+, Pandas, Pydantic
* **Interface e gráficos:** Streamlit
* **LLM e contingência:** Google GenAI SDK (modelo configurável em `MODELO_GEMINI`) + motor heurístico local (Demo Mode)

---

## ⚙️ Como Executar Localmente

```bash
git clone https://github.com/SEU-USUARIO/smartops-ai-logistics-agent.git
cd smartops-ai-logistics-agent
pip install -r requirements.txt
streamlit run app.py
```

Para usar o LLM real, informe uma chave do Google AI Studio na barra lateral. Sem chave, o app roda no modo demonstração.

---

## ⚠️ Limitações do Protótipo
* **Histórico em memória:** os chamados e os indicadores da Aba 2 ficam em `st.session_state` e partem de registros de demonstração.
* **Webhook simulado:** a Aba 3 gera e exibe o payload validado, mas não dispara requisições HTTP externas.
* **Modo demonstração:** sem chave do Gemini, o app usa um motor de regras local para permitir testar a interface e os guardrails.
* **Sem avaliação formal:** ainda não há um conjunto de casos de teste medindo a taxa de acerto do modelo.
