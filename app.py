import time
from datetime import datetime
from typing import Literal

import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field

# Tentativa de importação do SDK oficial Google GenAI
try:
    from google import genai
    from google.genai import types

    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# Nome do modelo em um só lugar (confira o nome atual no Google AI Studio)
MODELO_GEMINI = "gemini-3.6-flash"

# ==========================================
# CONFIGURAÇÃO DA PÁGINA
# ==========================================
st.set_page_config(
    page_title="SmartOps AI | Triagem Logística & Reembolsos",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================
# SCHEMA DE DADOS (STRUCTURED OUTPUTS)
# ==========================================
class TicketTriageSchema(BaseModel):
    categoria_falha: Literal[
        "Extravio Logístico",
        "Atraso Crítico de Entrega",
        "Item Faltante ou Incorreto",
        "Violação de Embalagem / Qualidade",
        "Suspeita de Fraude / Abuso",
    ] = Field(description="Categoria operacional primária do incidente.")
    responsabilidade_primaria: Literal[
        "Malha Logística (Entregador)",
        "Parceiro (Restaurante/Loja)",
        "Plataforma / App",
        "Cliente / Inconclusivo",
    ] = Field(description="Entidade responsável pelo atrito operacional.")
    severidade: Literal["Baixa", "Média", "Alta", "Crítica"] = Field(
        description="Nível de urgência operacional do ticket."
    )
    sentimento_cliente: Literal[
        "Neutro", "Frustrado", "Irritado", "Crítico (Ameaça Legal/Procon)"
    ] = Field(description="Estado emocional detectado no relato.")
    risco_churn_percentual: int = Field(
        ge=0, le=100, description="Probabilidade estimada de perda do cliente (0 a 100)."
    )
    score_risco_fraude: int = Field(
        ge=0, le=100, description="Indicador de possível abuso de política de reembolso (0 a 100)."
    )
    acao_financeira_recomendada: Literal[
        "Reembolso Total Automático",
        "Reembolso Parcial + Cupom de Retenção",
        "Apenas Cupom de Desculpas",
        "Escalar para Auditoria Humana (Bloqueio Preventivo)",
    ] = Field(description="Decisão financeira baseada em política de risco.")
    valor_reembolso_aprovado: float = Field(
        description="Valor financeiro em R$ sugerido para estorno ou crédito imediato."
    )
    justificativa_executiva: str = Field(
        description="Explicação técnica concisa do racional de decisão do agente."
    )
    mensagem_resposta_cliente: str = Field(
        description="Mensagem empática, personalizada e resolutiva pronta para envio no chat."
    )


# ==========================================
# ESTADO DA SESSÃO (HISTÓRICO EM MEMÓRIA)
# ==========================================
if "historico_tickets" not in st.session_state:
    st.session_state.historico_tickets = [
        {
            "id_pedido": "ORD-98412",
            "horario": "19:42",
            "valor_pedido": 84.90,
            "valor_aprovado": 84.90,
            "categoria": "Extravio Logístico",
            "responsabilidade": "Malha Logística (Entregador)",
            "severidade": "Crítica",
            "churn_risk": 88,
            "acao": "Reembolso Total Automático",
            "latencia_s": 1.2,
        },
        {
            "id_pedido": "ORD-98419",
            "horario": "20:05",
            "valor_pedido": 49.50,
            "valor_aprovado": 18.00,
            "categoria": "Item Faltante ou Incorreto",
            "responsabilidade": "Parceiro (Restaurante/Loja)",
            "severidade": "Média",
            "churn_risk": 45,
            "acao": "Reembolso Parcial + Cupom de Retenção",
            "latencia_s": 0.9,
        },
        {
            "id_pedido": "ORD-98455",
            "horario": "20:31",
            "valor_pedido": 142.00,
            "valor_aprovado": 0.00,
            "categoria": "Suspeita de Fraude / Abuso",
            "responsabilidade": "Cliente / Inconclusivo",
            "severidade": "Alta",
            "churn_risk": 30,
            "acao": "Escalar para Auditoria Humana (Bloqueio Preventivo)",
            "latencia_s": 1.4,
        },
    ]

# ==========================================
# MOTOR DE DECISÃO & GUARDRAILS DETERMINÍSTICOS
# ==========================================
SYSTEM_PROMPT = """
Você é um Agente Sênior de Operações Logísticas e Prevenção a Perdas de uma plataforma líder de Food Delivery.
Sua função é analisar reclamações de clientes, cruzar com os metadados do pedido e tomar decisões de reembolso e retenção.

DIRETRIZES E GUARDRAILS DE NEGÓCIO:
1. Se a conta do cliente tiver histórico elevado de reembolsos recentes (>= 4 nos últimos 30 dias) e o valor for alto, aumente o score_risco_fraude e recomende "Escalar para Auditoria Humana (Bloqueio Preventivo)" com valor_reembolso_aprovado = 0.0.
2. Se o cliente for VIP/Recorrente sem histórico de abuso e houver extravio ou atraso > 45 min, aprove reembolso integral ou parcial generoso para evitar churn.
3. Se faltar apenas 1 item secundário (ex: refrigerante ou acompanhamento), recomende "Reembolso Parcial + Cupom de Retenção" proporcional (~25% a 40% do pedido).
4. A mensagem ao cliente deve ser humana, direta, sem jargões robóticos, assumindo a resolução imediatamente.
"""

ACAO_ESCALAR = "Escalar para Auditoria Humana (Bloqueio Preventivo)"


def aplicar_guardrails(
    r: TicketTriageSchema, valor_pedido: float, reembolsos_30d: int
) -> tuple[TicketTriageSchema, list[str]]:
    """Regras determinísticas em Python que valem mesmo se o LLM falhar ou alucinar."""
    r = r.model_copy()
    ajustes = []

    if reembolsos_30d >= 4 and valor_pedido > 80 and r.acao_financeira_recomendada != ACAO_ESCALAR:
        r.acao_financeira_recomendada = ACAO_ESCALAR
        r.score_risco_fraude = max(r.score_risco_fraude, 80)
        r.mensagem_resposta_cliente = (
            "Olá! Recebemos sua solicitação e nossa equipe já assumiu o caso "
            "com prioridade. Você receberá uma atualização em breve."
        )
        ajustes.append("forçado escalonamento por histórico de reembolsos")

    if r.valor_reembolso_aprovado > valor_pedido:
        r.valor_reembolso_aprovado = valor_pedido
        ajustes.append("valor limitado ao total do pedido")

    if r.valor_reembolso_aprovado < 0:
        r.valor_reembolso_aprovado = 0.0
        ajustes.append("valor negativo corrigido")

    if r.acao_financeira_recomendada == ACAO_ESCALAR and r.valor_reembolso_aprovado != 0:
        r.valor_reembolso_aprovado = 0.0
        ajustes.append("escalonamento não aprova valor")

    r.valor_reembolso_aprovado = round(r.valor_reembolso_aprovado, 2)
    return r, ajustes


def executar_heuristica_fallback(
    relato: str,
    valor_pedido: float,
    atraso_min: int,
    reembolsos_30d: int,
    cliente_vip: bool,
) -> TicketTriageSchema:
    """Motor determinístico de contingência caso a API Key não seja informada."""
    texto = relato.lower()

    if reembolsos_30d >= 4 and valor_pedido > 80:
        return TicketTriageSchema(
            categoria_falha="Suspeita de Fraude / Abuso",
            responsabilidade_primaria="Cliente / Inconclusivo",
            severidade="Alta",
            sentimento_cliente="Irritado",
            risco_churn_percentual=35,
            score_risco_fraude=85,
            acao_financeira_recomendada=ACAO_ESCALAR,
            valor_reembolso_aprovado=0.0,
            justificativa_executiva=(
                f"Conta apresenta {reembolsos_30d} solicitações de estorno em 30 dias em ticket alto (R$ {valor_pedido:.2f}). "
                "Guardrail de risco ativado para validação de evidência."
            ),
            mensagem_resposta_cliente=(
                "Olá! Recebemos sua solicitação. Como identificamos uma divergência no registro de entrega deste pedido, "
                "nossa equipe especializada já assumiu o caso com prioridade."
            ),
        )

    if (
        any(k in texto for k in ["sumiu", "não chegou", "nao chegou", "extraviado", "finalizou sem entregar"])
        or atraso_min >= 60
    ):
        return TicketTriageSchema(
            categoria_falha="Extravio Logístico",
            responsabilidade_primaria="Malha Logística (Entregador)",
            severidade="Crítica",
            sentimento_cliente="Crítico (Ameaça Legal/Procon)" if "procon" in texto else "Irritado",
            risco_churn_percentual=90 if cliente_vip else 78,
            score_risco_fraude=12,
            acao_financeira_recomendada="Reembolso Total Automático",
            valor_reembolso_aprovado=round(valor_pedido, 2),
            justificativa_executiva=(
                f"Quebra severa de SLA ({atraso_min} min de atraso) com indício de falha na conclusão da rota. "
                "Estorno integral automatizado para retenção imediata."
            ),
            mensagem_resposta_cliente=(
                f"Sinto muito por essa experiência. Já confirmei a falha na rota da sua entrega e processei o reembolso integral "
                f"de R$ {valor_pedido:.2f} na sua carteira."
            ),
        )

    if any(k in texto for k in ["faltou", "faltando", "veio errado", "sem refrigerante", "incompleto", "esqueceu"]):
        estorno_parcial = round(valor_pedido * 0.35, 2)
        return TicketTriageSchema(
            categoria_falha="Item Faltante ou Incorreto",
            responsabilidade_primaria="Parceiro (Restaurante/Loja)",
            severidade="Média",
            sentimento_cliente="Frustrado",
            risco_churn_percentual=48,
            score_risco_fraude=15,
            acao_financeira_recomendada="Reembolso Parcial + Cupom de Retenção",
            valor_reembolso_aprovado=estorno_parcial,
            justificativa_executiva=(
                "Falha de conferência na expedição do restaurante parceiro. Aplicado estorno proporcional de 35%."
            ),
            mensagem_resposta_cliente=(
                f"Poxa, sentimos muito que seu pedido tenha chegado incompleto! Já creditamos R$ {estorno_parcial:.2f} "
                "referentes ao item afetado diretamente no seu saldo."
            ),
        )

    estorno_padrao = round(valor_pedido * 0.50, 2)
    return TicketTriageSchema(
        categoria_falha="Violação de Embalagem / Qualidade",
        responsabilidade_primaria="Malha Logística (Entregador)",
        severidade="Alta",
        sentimento_cliente="Frustrado",
        risco_churn_percentual=65,
        score_risco_fraude=20,
        acao_financeira_recomendada="Reembolso Parcial + Cupom de Retenção",
        valor_reembolso_aprovado=estorno_padrao,
        justificativa_executiva=(
            "Avaria física durante o deslocamento ou atraso moderado. Resolução automática via compensação financeira parcial."
        ),
        mensagem_resposta_cliente=(
            f"Pedimos desculpas pelo ocorrido com o seu pedido. Liberamos imediatamente um reembolso de R$ {estorno_padrao:.2f}."
        ),
    )


def analisar_ticket_com_ia(
    api_key: str,
    relato: str,
    id_pedido: str,
    valor_pedido: float,
    atraso_min: int,
    reembolsos_30d: int,
    cliente_vip: bool,
) -> tuple[TicketTriageSchema, str]:
    """Executa inferência estruturada via Gemini + camada de guardrails em Python."""
    if api_key and GENAI_AVAILABLE:
        try:
            client = genai.Client(api_key=api_key)
            prompt_usuario = f"""
            DADOS DO PEDIDO:
            - ID: {id_pedido}
            - Valor Total: R$ {valor_pedido:.2f}
            - Tempo de Atraso além do SLA: {atraso_min} minutos
            - Reembolsos solicitados nos últimos 30 dias: {reembolsos_30d}
            - Cliente Assinante/VIP: {"Sim" if cliente_vip else "Não"}

            RELATO BRUTO DO CLIENTE:
            "{relato}"
            """
            response = client.models.generate_content(
                model=MODELO_GEMINI,
                contents=prompt_usuario,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=TicketTriageSchema,
                ),
            )
            resultado = TicketTriageSchema.model_validate_json(response.text)
            resultado, ajustes = aplicar_guardrails(resultado, valor_pedido, reembolsos_30d)
            motor = f"{MODELO_GEMINI} (Structured JSON API)"
            if ajustes:
                motor += " + Guardrail Python (" + "; ".join(ajustes) + ")"
            return resultado, motor
        except Exception as e:
            st.warning(f"Aviso na API externa ({e}). Acionando motor de contingência local.")

    time.sleep(0.5)
    resultado_fallback = executar_heuristica_fallback(
        relato, valor_pedido, atraso_min, reembolsos_30d, cliente_vip
    )
    resultado_fallback, ajustes = aplicar_guardrails(resultado_fallback, valor_pedido, reembolsos_30d)
    motor = "Motor de Regras & Heurística Local (Demo Mode)"
    if ajustes:
        motor += " + Guardrail Python (" + "; ".join(ajustes) + ")"
    return resultado_fallback, motor


# ==========================================
# SIDEBAR — CONFIGURAÇÃO (KPIs são preenchidos no fim do arquivo)
# ==========================================
with st.sidebar:
    st.title("⚙️ SmartOps AI")
    st.caption("Orquestrador de Suporte & Logística")

    api_key_input = st.text_input(
        "🔑 Gemini API Key (Opcional)",
        type="password",
        help="Insira sua chave do Google AI Studio para inferência LLM real, ou deixe em branco para usar o Modo Simulação.",
    )

    st.divider()
    sidebar_kpis = st.container()

    st.divider()
    st.markdown(
        "**Stack:**\n"
        "- `Python` + `Streamlit`\n"
        "- `Google GenAI SDK` + `Pydantic`\n"
        "- Payload JSON p/ `n8n`"
    )

# ==========================================
# CORPO PRINCIPAL DO APLICATIVO
# ==========================================
st.title("🚀 SmartOps AI — Triagem Logística, Reembolsos & Retenção")
st.markdown(
    "Agente para **classificação de incidentes de entrega, decisão financeira com guardrails antifraude "
    "e geração de payload para n8n**."
)

tab_triagem, tab_dashboard, tab_arquitetura = st.tabs(
    ["⚡ Triagem de Chamado", "📈 Analytics & Controle de Perdas", "🧩 Payload n8n & Arquitetura"]
)

# ------------------------------------------
# ABA 1: TRIAGEM
# ------------------------------------------
with tab_triagem:
    col_input, col_output = st.columns([1, 1.15], gap="large")

    with col_input:
        st.subheader("📥 Entrada do Ticket")

        preset = st.selectbox(
            "Carregar cenário de teste rápido:",
            [
                "Cenário 1: Extravio crítico + Cliente VIP irritado",
                "Cenário 2: Item faltante no pedido (Parcial)",
                "Cenário 3: Tentativa suspeita de fraude recorrente",
                "Digitar relato personalizado...",
            ],
        )

        if preset == "Cenário 1: Extravio crítico + Cliente VIP irritado":
            default_id = "ORD-99104"
            default_val = 118.90
            default_delay = 68
            default_ref = 0
            default_vip = True
            default_text = (
                "Meu pedido consta como entregue no aplicativo há 20 minutos, mas o entregador não apareceu na portaria! "
                "Estou esperando há mais de 1 hora, sou assinante do clube e quero meu estorno de R$ 118,90 agora ou vou abrir reclamação no Procon!"
            )
        elif preset == "Cenário 2: Item faltante no pedido (Parcial)":
            default_id = "ORD-99215"
            default_val = 64.00
            default_delay = 10
            default_ref = 1
            default_vip = True
            default_text = (
                "O lanche chegou certinho, mas o restaurante esqueceu de mandar os dois refrigerantes lata e a batata grande que estavam pagos na nota."
            )
        elif preset == "Cenário 3: Tentativa suspeita de fraude recorrente":
            default_id = "ORD-99380"
            default_val = 189.00
            default_delay = 5
            default_ref = 6
            default_vip = False
            default_text = (
                "Comida veio toda revirada e fria, impossível comer. Quero o reembolso total do pedido no meu Pix imediatamente."
            )
        else:
            default_id = "ORD-99500"
            default_val = 75.00
            default_delay = 35
            default_ref = 0
            default_vip = False
            default_text = ""

        c1, c2 = st.columns(2)
        id_pedido = c1.text_input("ID do Pedido", value=default_id)
        valor_pedido = c2.number_input(
            "Valor do Pedido (R$)", min_value=10.0, max_value=1000.0, value=default_val, step=5.0
        )

        c3, c4 = st.columns(2)
        atraso_min = c3.slider("Atraso além do SLA (min)", 0, 120, default_delay)
        reembolsos_30d = c4.number_input(
            "Reembolsos (últimos 30 dias)", min_value=0, max_value=20, value=default_ref
        )

        cliente_vip = st.checkbox("Cliente Assinante Clube / Alta Frequência (LTV Alto)", value=default_vip)

        relato_cliente = st.text_area(
            "Mensagem Bruta do Cliente (Chat / SAC):",
            value=default_text,
            height=130,
            placeholder="Cole aqui a reclamação enviada pelo cliente no app...",
        )

        executar_btn = st.button(
            "⚡ Executar Agente de Triagem & Decisão", type="primary", width="stretch"
        )

    with col_output:
        st.subheader("🧠 Diagnóstico & Decisão do Agente")

        if executar_btn:
            if not relato_cliente.strip():
                st.error("Por favor, insira o relato do cliente para processar a triagem.")
            else:
                t0 = time.time()
                with st.spinner("Analisando o pedido, o histórico antifraude e gerando a decisão..."):
                    resultado, motor_usado = analisar_ticket_com_ia(
                        api_key=api_key_input,
                        relato=relato_cliente,
                        id_pedido=id_pedido,
                        valor_pedido=valor_pedido,
                        atraso_min=atraso_min,
                        reembolsos_30d=reembolsos_30d,
                        cliente_vip=cliente_vip,
                    )
                latencia = round(time.time() - t0, 2)

                # Salva no histórico da sessão
                novo_registro = {
                    "id_pedido": id_pedido,
                    "horario": datetime.now().strftime("%H:%M"),
                    "valor_pedido": valor_pedido,
                    "valor_aprovado": resultado.valor_reembolso_aprovado,
                    "categoria": resultado.categoria_falha,
                    "responsabilidade": resultado.responsabilidade_primaria,
                    "severidade": resultado.severidade,
                    "churn_risk": resultado.risco_churn_percentual,
                    "acao": resultado.acao_financeira_recomendada,
                    "latencia_s": latencia,
                }
                st.session_state.historico_tickets.insert(0, novo_registro)
                st.session_state.ultimo_payload = {
                    "webhook_event": "logistics.triage.completed",
                    "order_id": id_pedido,
                    "timestamp": datetime.now().isoformat(),
                    "engine": motor_usado,
                    "latency_seconds": latencia,
                    "decision": resultado.model_dump(),
                }

                st.caption(f"⚙️ Processado em **{latencia}s** via `{motor_usado}`")

                m1, m2, m3 = st.columns(3)
                m1.metric("Severidade", resultado.severidade)
                m2.metric("Risco de Churn", f"{resultado.risco_churn_percentual}%")
                m3.metric("Reembolso Aprovado", f"R$ {resultado.valor_reembolso_aprovado:.2f}")

                st.info(
                    f"**Ação Recomendada:** {resultado.acao_financeira_recomendada}\n\n"
                    f"**Categoria:** {resultado.categoria_falha} | **Responsável:** {resultado.responsabilidade_primaria}\n\n"
                    f"**Racional Executivo:** {resultado.justificativa_executiva}"
                )

                st.markdown("**💬 Resposta Pronta para o Chat do Cliente:**")
                st.code(resultado.mensagem_resposta_cliente, language="text")
        else:
            st.info("Selecione um cenário ao lado e clique em **Executar Agente** para ver a decisão estruturada.")

# ------------------------------------------
# ABA 2: DASHBOARD & CONTROLE DE PERDAS
# ------------------------------------------
with tab_dashboard:
    st.subheader("📊 Painel Operacional e Financeiro (Dados de Demonstração)")
    df = pd.DataFrame(st.session_state.historico_tickets)

    k1, k2, k3, k4 = st.columns(4)
    valor_total_pedidos = df["valor_pedido"].sum()
    valor_total_estornado = df["valor_aprovado"].sum()
    sob_auditoria = df.loc[df["acao"] == ACAO_ESCALAR, "valor_pedido"].sum()

    k1.metric("Volume Analisado (R$)", f"R$ {valor_total_pedidos:,.2f}")
    k2.metric("Estornos Liberados (R$)", f"R$ {valor_total_estornado:,.2f}")
    k3.metric("Sob Auditoria Humana (R$)", f"R$ {sob_auditoria:,.2f}")
    k4.metric("Risco Médio de Churn", f"{df['churn_risk'].mean():.1f}%")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("**Distribuição por Categoria de Incidente**")
        contagem_cat = df["categoria"].value_counts()
        if not contagem_cat.empty:
            st.bar_chart(contagem_cat)

    with col_g2:
        st.markdown("**Alocação de Responsabilidade Operacional**")
        contagem_resp = df["responsabilidade"].value_counts()
        if not contagem_resp.empty:
            st.bar_chart(contagem_resp)

    st.markdown("**📋 Log de Decisões na Sessão**")
    st.dataframe(df, width="stretch", hide_index=True)

# ------------------------------------------
# ABA 3: PAYLOAD N8N & ARQUITETURA
# ------------------------------------------
with tab_arquitetura:
    st.subheader("🔗 Payload JSON para n8n")
    st.markdown(
        "Cada decisão do agente gera um payload JSON validado via `Pydantic`, pronto para ser consumido "
        "por um webhook no **n8n** (estorno no gateway de pagamento, ticket de auditoria humana ou alerta no Slack). "
        "Neste protótipo o payload é apenas gerado e exibido; nenhuma requisição externa é disparada."
    )

    payload_exibir = st.session_state.get(
        "ultimo_payload",
        {
            "webhook_event": "logistics.triage.completed",
            "order_id": "ORD-98412",
            "status": "Aguardando a primeira execução na Aba 1...",
        },
    )
    st.json(payload_exibir)

# ==========================================
# KPIs DA SIDEBAR (no fim, para refletir o ticket recém-processado)
# ==========================================
with sidebar_kpis:
    st.subheader("📊 Métricas da Sessão (Demo)")
    df_hist = pd.DataFrame(st.session_state.historico_tickets)
    total_processados = len(df_hist)
    latencia_media = df_hist["latencia_s"].mean() if total_processados > 0 else 0.0
    automacao_rate = (
        (df_hist["acao"] != ACAO_ESCALAR).mean() * 100 if total_processados > 0 else 0.0
    )
    st.metric("Chamados Processados", f"{total_processados}")
    st.metric("Latência Média", f"{latencia_media:.2f}s")
    st.metric("Resolução Autônoma", f"{automacao_rate:.0f}%")
