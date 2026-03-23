#include <cekeikon.h>

const COR BRANCO(255, 255, 255);
const COR PRETO(0, 0, 0);
const COR CIANO(255, 255, 0);
const COR VERMELHO(0, 0, 255);
const COR AZUL(0, 255, 0);
const COR VERDE(255, 0, 0);

bool dentro(const Mat_<COR>& imagem, const Point& p)
{
    return 0 <= p.y && p.y < imagem.rows && 0 <= p.x && p.x < imagem.cols;
}

bool temCor(const Mat_<COR>& imagem, const Point& p, const COR& cor)
{
    return dentro(imagem, p) && distancia(imagem(p.y, p.x), cor) == 0;
}

void adicionaVizinhos(queue<Point>& fila, const Point& p)
{
    fila.push(Point(p.x, p.y - 1));
    fila.push(Point(p.x, p.y + 1));
    fila.push(Point(p.x - 1, p.y));
    fila.push(Point(p.x + 1, p.y));
}

void pintaComponente(Mat_<COR>& imagem, int linha, int coluna, const COR& origem, const COR& destino)
{
    queue<Point> fila;
    fila.push(Point(coluna, linha));

    while (!fila.empty())
    {
        Point atual = fila.front();
        fila.pop();

        if (!temCor(imagem, atual, origem))
            continue;

        imagem(atual.y, atual.x) = destino;
        adicionaVizinhos(fila, atual);
    }
}

COR corPorQuantidadeDeBuracos(int indice)
{
    if (indice == 0)
        return VERMELHO;
    if (indice == 1)
        return AZUL;
    return VERDE;
}

void pintaFundo(Mat_<COR>& imagem)
{
    pintaComponente(imagem, 0, 0, BRANCO, CIANO);
}

COR escolheCorDoNumero(Mat_<COR>& imagem, int linha, int coluna)
{
    int quantidadeDeBuracos = 0;
    queue<Point> fila;
    fila.push(Point(coluna, linha));

    while (!fila.empty())
    {
        Point atual = fila.front();
        fila.pop();

        if (temCor(imagem, atual, BRANCO))
        {
            pintaComponente(imagem, atual.y, atual.x, BRANCO,
                            corPorQuantidadeDeBuracos(quantidadeDeBuracos));
            quantidadeDeBuracos++;
            continue;
        }

        if (!temCor(imagem, atual, PRETO))
            continue;

        imagem(atual.y, atual.x) = CIANO;
        adicionaVizinhos(fila, atual);
    }

    return corPorQuantidadeDeBuracos(quantidadeDeBuracos);
}

int main()
{
    Mat_<COR> imagem;
    le(imagem, "assets/c2.bmp");

    Mat_<COR> fundoColorido = imagem.clone();
    Mat_<COR> resultado = imagem.clone();

    pintaFundo(fundoColorido);

    for (int l = 0; l < imagem.rows; l++)
    {
        for (int c = 0; c < imagem.cols; c++)
        {
            if (!temCor(fundoColorido, Point(c, l), PRETO))
                continue;

            COR corDoNumero = escolheCorDoNumero(fundoColorido, l, c);
            pintaComponente(resultado, l, c, PRETO, corDoNumero);
        }
    }

    imp(resultado, "results/c2_colorido.png");
}
