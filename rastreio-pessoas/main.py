import cv2
import numpy as np

# Caminhos dos arquivos
ARQUIVO_VIDEO = 'rastreio-pessoas/walking.mp4'
ARQUIVO_MODELO = 'rastreio-pessoas/frozen_inference_graph.pb'
ARQUIVO_CFG = 'rastreio-pessoas/ssd_mobilenet_v2_coco.pbtxt'

# Variáveis globais para contar pessoas únicas
total_pessoas = 0
pessoas_detectadas = []  # Lista para registrar pessoas já detectadas

def carregar_modelo(ARQUIVO_MODELO, ARQUIVO_CFG):
    '''
    Carrega o modelo de deep learning do TensorFlow para detecção de objetos.
    ARQUIVO_MODELO: Caminho para o arquivo .pb contendo os pesos do modelo.
    ARQUIVO_CFG: Caminho para o arquivo .pbtxt contendo a configuração do modelo.
    Retorna o modelo carregado.
    '''
    try:
        modelo = cv2.dnn.readNetFromTensorflow(ARQUIVO_MODELO, ARQUIVO_CFG)
    except cv2.error as erro:
        print(f"Erro ao carregar o modelo: {erro}")
        exit()
    return modelo

def aplicar_supressao_nao_maxima(caixas, confiancas, limiar_conf, limiar_supr):
    '''
    Aplica a Supressão Não Máxima para reduzir o número de caixas delimitadoras sobrepostas.
    caixas: Lista de caixas delimitadoras.
    confiancas: Lista de confianças de cada caixa.
    limiar_conf: Limiar de confiança para considerar detecções.
    limiar_supr: Limiar de sobreposição para suprimir caixas redundantes.
    Retorna uma lista de caixas após aplicar a supressão.
    '''
    indices = cv2.dnn.NMSBoxes(caixas, confiancas, limiar_conf, limiar_supr)
    return [caixas[i] for i in indices.flatten()] if len(indices) > 0 else []

def pessoa_detectada(nova_caixa, limiar_sobreposicao):
    '''
    Verifica se uma nova caixa delimitadora de pessoa já foi detectada.
    Retorna True se a pessoa já foi detectada, False caso contrário.
    '''
    for caixa in pessoas_detectadas:
        (cx, cy, cw, ch) = caixa
        (nx, ny, nw, nh) = nova_caixa

        # Calcula a área de sobreposição
        x_overlap = max(0, min(cx + cw, nx + nw) - max(cx, nx))
        y_overlap = max(0, min(cy + ch, ny + nh) - max(cy, ny))
        area_intersecao = x_overlap * y_overlap
        area_caixa_atual = cw * ch
        area_nova_caixa = nw * nh

        # Calcula a razão de sobreposição
        razao_sobreposicao = area_intersecao / min(area_caixa_atual, area_nova_caixa)

        if razao_sobreposicao > limiar_sobreposicao:
            return True

    return False

def desenhar_quadrante(frame, caixas_finais):
    '''
    Desenha um quadrante em volta de cada pessoa detectada.
    frame: Frame de vídeo onde desenhar o quadrante.
    caixas_finais: Lista de caixas delimitadoras das pessoas detectadas.
    '''
    for (inicioX, inicioY, largura, altura) in caixas_finais:
        cv2.rectangle(frame, (inicioX, inicioY), (inicioX + largura, inicioY + altura), (0, 255, 0), 2)
        cv2.circle(frame, (inicioX + largura // 2, inicioY + altura // 2), 5, (0, 0, 255), -1)
        cv2.putText(frame, "Pessoa", (inicioX, inicioY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def main():
    global total_pessoas
    global pessoas_detectadas

    '''
    Função principal que executa o rastreio de pessoas no vídeo.
    '''
    captura = cv2.VideoCapture(ARQUIVO_VIDEO)
    detector_pessoas = carregar_modelo(ARQUIVO_MODELO, ARQUIVO_CFG)
    pausado = False

    while True:
        if not pausado:
            ret, frame = captura.read()
            if not ret:
                break

            # Criação do blob a partir do frame e realização da detecção
            blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)
            detector_pessoas.setInput(blob)
            deteccoes = detector_pessoas.forward()

            caixas = []
            confiancas = []

            # Extração das caixas delimitadoras e confianças das detecções
            for i in range(deteccoes.shape[2]):
                confianca = deteccoes[0, 0, i, 2]
                if confianca > 0.5:
                    (altura, largura) = frame.shape[:2]
                    caixa = deteccoes[0, 0, i, 3:8] * np.array([largura, altura, largura, altura])
                    (inicioX, inicioY, fimX, fimY) = caixa.astype("int")

                    nova_caixa = (inicioX, inicioY, fimX - inicioX, fimY - inicioY)

                    if not pessoa_detectada(nova_caixa, limiar_sobreposicao=0.3):
                        caixas.append([inicioX, inicioY, fimX - inicioX, fimY - inicioY])
                        confiancas.append(float(confianca))
                        pessoas_detectadas.append(nova_caixa)
                        total_pessoas += 1  # Incrementa o contador de pessoas únicas

            # Aplicação da supressão não máxima para finalizar as caixas delimitadoras
            caixas_finais = aplicar_supressao_nao_maxima(caixas, confiancas, limiar_conf=0.5, limiar_supr=0.4)

            # Desenho dos quadrantes em volta das pessoas detectadas
            desenhar_quadrante(frame, caixas_finais)

            # Exibição do número de pessoas detectadas
            cv2.putText(frame, f"Total de pessoas: {total_pessoas}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Exibição do frame processado e controle de pausa/play
        cv2.imshow("Rastreio de Pessoas", frame)
        
        tecla = cv2.waitKey(30) & 0xFF
        if tecla == ord('q'):
            break
        elif tecla == ord('p'):
            pausado = not pausado

    # Liberação dos recursos ao finalizar
    captura.release()
    cv2.destroyAllWindows()

    # Exibi total de pessoas únicas detectadas
    print(f"Total de pessoas únicas detectadas atravessando a rua: {total_pessoas}")

if __name__ == "__main__":
    main()
