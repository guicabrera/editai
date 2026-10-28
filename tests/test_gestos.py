"""Os gestos do personagem: o gesto do tamanho da frase (entrada, sustentação, saída) e a
boca pelas sílabas, colada no nariz."""

import json

import numpy as np
import pytest
from PIL import Image

from editor.montagem import (
    CamadaDeGestos,
    Clipe,
    Montagem,
    PersonagemInvalido,
    ler_gestos,
    trilha_de_gestos,
)

FPS = 10


def _clipe(cor: int, n: int = 10, entrada: int = 2, sustenta=(2, 6), fim: int = 8) -> Clipe:
    quadros = []
    for k in range(n):
        q = np.zeros((20, 20, 4), np.uint8)
        q[..., 3] = 255
        q[..., 0], q[..., 1] = cor, k
        quadros.append(q)
    return Clipe(quadros, [(10, 5)] * n, entrada, sustenta, fim)


def _bocas():
    def spr(v):
        s = np.zeros((2, 2, 4), np.uint8)
        s[...] = (v, v, v, 255)
        return s, -1, 3
    return {"fechada": spr(1), "meia": spr(2), "aberta": spr(3)}


def _camada(trilha):
    clipes = {"ouvindo": _clipe(10), "apontando": _clipe(30), "acenando": _clipe(40)}
    return CamadaDeGestos(clipes, _bocas(), FPS, trilha)


def test_trilha_mapeia_as_poses_e_fecha_com_aceno():
    clipes = {"ouvindo": 1, "apontando": 1, "acenando": 1}
    poses = [(0, 2, "neutro"), (2, 4, "apontando_lado"), (4, 6, "pensando"), (6, 8, "neutro")]
    assert trilha_de_gestos(poses, clipes) == [
        (0, 2, None),            # o áudio já começa falando: sem aceno
        (2, 4, "apontando"),
        (4, 6, None),            # sem o clipe do pensando
        (6, 8, "acenando")]


def test_gesto_entra_sustenta_e_sai_no_fim_da_sustentacao():
    # entrada 0,2 s; sustentação de 4 quadros (0,4 s por meia-volta); saída 0,2 s.
    # A frase tem 2 s: o número de meias-voltas é ímpar, para sair do fim da sustentação
    c = _camada([(0, 2, "apontando")])
    (t,) = c.toques
    assert t.voltas % 2 == 1
    assert c._quadro_do_gesto(0.1)[1] == 1                 # entrando
    assert c._quadro_do_gesto(0.2)[1] == 2                 # começo da sustentação
    _e, s, fim = t.tempos(c.clipes["apontando"], FPS)
    assert c._quadro_do_gesto(s - 0.01)[1] == 5            # chegando no fim da sustentação…
    assert c._quadro_do_gesto(s + 0.01)[1] == 6            # …a saída começa nele
    assert c._quadro_do_gesto(s + 0.11)[1] == 7
    assert c._quadro_do_gesto(fim + 0.01)[0] is c.clipes["ouvindo"]


def test_frases_seguidas_com_o_mesmo_gesto_viram_um_toque():
    c = _camada([(0, 2, "apontando"), (2, 4, "apontando"), (4, 9, None)])
    assert len(c.toques) == 1
    assert c.toques[0].tempos(c.clipes["apontando"], FPS)[2] > 3


def test_gestos_nao_se_sobrepoem():
    c = _camada([(0, 1, "apontando"), (1, 1.2, "acenando"), (1.2, 1.5, "apontando"),
                 (1.5, 9, "acenando")])
    assert c.toques[0].gesto == "apontando"
    assert all(b.inicio >= a.tempos(c.clipes[a.gesto], FPS)[2] - 1e-9
               for a, b in zip(c.toques, c.toques[1:]))


def test_boca_das_silabas_colada_no_nariz():
    c = _camada([(0, 5, None)])
    c.bocas, c.fps_do_video = ["fechada", "aberta"], 1.0
    q0, q1 = c.rgba_falando(0.0), c.rgba_falando(1.0)
    # nariz em (10, 5), boca 1 px à esquerda e 3 abaixo
    assert q0[8, 9, 0] == 1 and q1[8, 9, 0] == 3
    assert q1[0, 0, 0] == 10                                # o resto é o quadro da base


def test_ler_gestos_da_pasta(tmp_path):
    for nome in ("gesto_ouvindo_semboca", "gesto_apontando_semboca"):
        Image.new("RGBA", (8, 8), (9, 9, 9, 255)).save(tmp_path / f"{nome}.gif")
    (tmp_path / "bocas").mkdir()
    for f in ("fechada", "meia", "aberta"):
        Image.new("RGBA", (2, 2), (1, 1, 1, 255)).save(tmp_path / "bocas" / f"boca_{f}.png")
    g = {"narizes": [[4, 2]], "entrada": [0, 0], "sustenta": [0, 0], "saida": [0, 0]}
    (tmp_path / "gestos.json").write_text(json.dumps({
        "fps": 12,
        "gestos": {"ouvindo": {**g, "arquivo": "gesto_ouvindo_semboca.gif"},
                   "apontando": {**g, "arquivo": "gesto_apontando_semboca.gif"}},
        "bocas": {f: {"arquivo": f"bocas/boca_{f}.png", "dx": -1, "dy": 2}
                  for f in ("fechada", "meia", "aberta")}}))
    clipes, bocas, fps = ler_gestos(tmp_path)
    assert set(clipes) == {"ouvindo", "apontando"}
    assert set(bocas) == {"fechada", "meia", "aberta"}
    assert fps == 12


def test_ler_gestos_sem_o_json(tmp_path):
    with pytest.raises(PersonagemInvalido):
        ler_gestos(tmp_path)


def test_gestos_vao_com_o_personagem(tmp_path):
    m = Montagem(tmp_path / "fundo.mp4", gestos=tmp_path)
    assert any("gestos" in e for e in m.problemas())


def test_despedida_tem_a_vez():
    # o apontando que começa em 3,5 s dura 0,8 s e ainda tocaria quando a última frase
    # começa (4 s): fica de fora, e o aceno entra no começo da frase dele
    c = _camada([(0, 3.5, None), (3.5, 4, "apontando"), (4, 6, "acenando")])
    assert [(t.inicio, t.gesto) for t in c.toques] == [(4, "acenando")]


def test_termos_entre_aspas_viram_dica_do_whisper():
    from editor.transcricao import termos_entre_aspas
    texto = ('Você sabe o que significa "whaddya"? Tipo uót du iu. Fica "whaddya". '
             'Exemplo: "Whaddya think?" Repete: “whaddya”!')
    assert termos_entre_aspas(texto) == ["whaddya", "Whaddya think"]
