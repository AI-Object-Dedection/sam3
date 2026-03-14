# Vibe Coding Rehberi

Bu dosya, AI araçlarıyla (Claude Code, Cursor, Copilot vb.) nasıl etkili çalışılacağını açıklar.

---

## Vibe Coding Nedir?

AI araçlarına projenin bağlamını vererek, birlikte kod geliştirmek. Siz yönlendirirsiniz, AI üretir ve açıklar. Anahtar: **AI'a ne kadar iyi bağlam verirseniz, o kadar iyi sonuç alırsınız.**

---

## Proje Dosya Hiyerarşisi (AI İçin)

AI agent'lar dosyaları şu sırayla okur:

```
1. CLAUDE.md          → Proje kuralları ve yapısı (her zaman okunur)
2. docs/03_...md      → Şu an hangi fazdayız, ne yapılmalı
3. docs/02_...md      → Teknik terimler ve kavramlar
4. src/config.py      → Proje ayarları
5. İlgili src/ dosyası → Üzerinde çalışılan dosya
```

**Önemli:** `CLAUDE.md` ve `docs/03_GELISTIRME_ADIMLARI.md` dosyalarını güncel tutun. AI agent'lar projeye baktığında en çok bu dosyalara güvenir.

---

## AI'a Nasıl Görev Verilir?

### İyi Görev Verme Örnekleri

```
✅ "dataset.py'de _create_mask fonksiyonu var. Bu fonksiyon şu an tüm
    sınıfları tek bir binary mask'a birleştiriyor. Bunu test etmek için
    scripts/ altına bir görselleştirme scripti yaz."

✅ "Faz 3'e geçiyoruz. docs/03_GELISTIRME_ADIMLARI.md'deki Faz 3
    yapılacak işler listesine bak ve ilk adımdan başla."

✅ "train.py'deki train_one_epoch fonksiyonunda TODO var. Forward pass
    ve loss hesaplama kısmını implement et. SAM3'ün çıktı yapısını
    bilmiyoruz, önce araştır."
```

### Kötü Görev Verme Örnekleri

```
❌ "Eğitimi yap" → Çok belirsiz, hangi faz, hangi adım?

❌ "Kodu düzelt" → Hangi dosya, hangi hata?

❌ "Her şeyi bir seferde bitir" → Çok büyük kapsam
```

### Görev Verme Şablonu

```
[Bağlam]:   Şu an Faz X'deyiz, Y dosyası üzerinde çalışıyoruz
[Görev]:    Z fonksiyonunu implement et / bu hatayı düzelt / şunu ekle
[Kısıtlar]: Basit tut / mevcut yapıyı bozma / Türkçe yorum yaz
[Test]:     Nasıl test edeceğimi de söyle
```

---

## MCP (Model Context Protocol) Sunucuları

### MCP Nedir?

MCP, AI agent'lara ek yetenekler kazandıran bağlantı noktalarıdır. Normalde AI sadece dosya okuyup yazabilir. MCP ile:
- Web'de arama yapabilir
- API'lere bağlanabilir
- Veritabanı sorgulayabilir
- Özel araçlar kullanabilir

### Bu Proje İçin Önerilen MCP Sunucuları

#### 1. Filesystem MCP (Zaten Dahil)
Claude Code'un varsayılan dosya sistemi erişimi. Ek kurulum gerekmez.

#### 2. Web Search / Brave Search MCP
**Ne işe yarar:** HuggingFace dokümantasyonu, PyTorch API'si, hata mesajları araştırma.

**Ne zaman lazım:**
- `Sam3Model`'in API'sini araştırırken
- Bir hata mesajının çözümünü ararken
- DACL10K ile ilgili güncel bilgi ararken

**Kurulum (Claude Code):**
```json
// ~/.claude/settings.json içine ekleyin:
{
  "mcpServers": {
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@anthropic-ai/mcp-brave-search"],
      "env": {
        "BRAVE_API_KEY": "your-api-key"
      }
    }
  }
}
```

#### 3. Sequential Thinking MCP
**Ne işe yarar:** Karmaşık problemleri adım adım düşünme. Özellikle hata ayıklama ve mimari kararlar için.

**Ne zaman lazım:**
- Eğitim döngüsü çalışmıyorsa
- Loss değeri düşmüyorsa
- Mimari karar verilecekse

**Kurulum (Claude Code):**
```json
{
  "mcpServers": {
    "sequential-thinking": {
      "command": "npx",
      "args": ["-y", "@anthropic-ai/mcp-sequential-thinking"]
    }
  }
}
```

#### 4. GitHub MCP
**Ne işe yarar:** GitHub işlemleri — PR oluşturma, issue yönetimi, kod inceleme.

**Ne zaman lazım:**
- Ekip üyeleri arasında kod paylaşırken
- PR açıp code review yaparken

**Kurulum (Claude Code):**
```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@anthropic-ai/mcp-github"],
      "env": {
        "GITHUB_TOKEN": "your-github-token"
      }
    }
  }
}
```

#### 5. HuggingFace MCP (Opsiyonel)
**Ne işe yarar:** Model kartlarını okuma, model bilgilerini çekme.

**Ne zaman lazım:** SAM3 modelinin detaylarını araştırırken.

### MCP Kurulumu — Adım Adım

1. Claude Code'u açın
2. Ayarlar dosyasını düzenleyin:
   - Windows: `%USERPROFILE%\.claude\settings.json`
   - macOS/Linux: `~/.claude/settings.json`
3. `mcpServers` bölümüne istediğiniz MCP'leri ekleyin
4. Claude Code'u yeniden başlatın
5. `/mcp` komutu ile aktif MCP'leri kontrol edin

### Öncelik Sırası

Hepsini birden kurmak zorunda değilsiniz. Önerilen sıra:

1. **Sequential Thinking** → Hemen kurun (karmaşık ML problemlerinde çok yardımcı)
2. **Web Search** → Hemen kurun (dokümantasyon araştırması için)
3. **GitHub** → Ekip çalışması başlayınca kurun
4. **HuggingFace** → Gerekirse kurun

---

## Vibe Coding İş Akışı

### Yeni Bir Göreve Başlarken

```
1. docs/03_GELISTIRME_ADIMLARI.md'yi aç
2. Hangi fazdayız kontrol et
3. O fazdaki yapılacak işlerden birini seç
4. AI'a görevi ver (şablon kullan)
5. AI'ın ürettiği kodu oku ve anla
6. Test et
7. Çalışıyorsa commit at
8. docs/03'teki checkbox'ı işaretle
```

### Hata Aldığınızda

```
1. Hata mesajını kopyala
2. AI'a yapıştır: "Bu hatayı alıyorum: [hata]. Ne yapmalıyım?"
3. AI'ın önerisini uygula
4. Hala çalışmıyorsa: "Daha basit bir yaklaşım dene"
5. Son çare: "Adım adım debug edelim, her adımda print koyalım"
```

### Her Oturum Sonunda

```
1. Nerede kaldığınızı docs/03'e not edin
2. Yarım kalan işleri TODO olarak bırakın
3. Commit atın: [faz-X] yapılan iş açıklaması
```

---

## Ekip İçi Çalışma

### Görev Dağılımı Önerisi

Her ekip üyesi bir AI agent ile çalışırken:
- **Kişi 1:** Veri seti tarafı (dataset.py, inspect_dataset.py)
- **Kişi 2:** Model tarafı (model.py, lora.py)
- **Kişi 3:** Eğitim tarafı (train.py, evaluate.py)

### Çakışma Önleme

- Herkes farklı dosyalar üzerinde çalışsın
- `config.py`'yi değiştirmeden önce ekiple konuşun (ortak dosya)
- Her iş bittiğinde hemen commit + push yapın

---

## Sık Yapılan Hatalar

| Hata | Çözüm |
|------|-------|
| AI çok karmaşık kod yazıyor | "Daha basit yaz, biz ML'de yeniyiz" de |
| AI tek seferde çok şey değiştiriyor | "Sadece X fonksiyonunu yaz, geri kalanına dokunma" de |
| AI İngilizce yorum yazıyor | "Yorumları Türkçe yaz" de |
| Hangi fazdayız bilmiyorum | `docs/03_GELISTIRME_ADIMLARI.md`'yi aç |
| AI'ın yazdığı kodu anlamıyorum | "Bu kodu satır satır açıkla" de |
| Bir şey çalışmıyor | Hata mesajını kopyalayıp AI'a ver |
| GPU bellek hatası | `config.py`'de `BATCH_SIZE`'ı 1'e düşür |

---

## Faydalı Claude Code Komutları

| Komut | Ne Yapar |
|-------|----------|
| `/help` | Yardım menüsü |
| `/clear` | Konuşma geçmişini temizle |
| `/compact` | Bağlamı sıkıştır (uzun konuşmalarda) |
| `/mcp` | Aktif MCP sunucularını göster |
| `/memory` | Kayıtlı bellek dosyalarını göster |

---

## Prompt Kütüphanesi

İşte bu projede sıkça kullanabileceğiniz hazır prompt'lar:

### Faz Başlatma
```
"docs/03_GELISTIRME_ADIMLARI.md'yi oku. Faz [X]'e başlıyoruz.
İlk yapılacak işten başla ve adım adım ilerle."
```

### Kod Açıklama
```
"[dosya_adı.py] dosyasındaki [fonksiyon_adı] fonksiyonunu
satır satır açıkla. Ben ML'de yeniyim."
```

### Hata Çözme
```
"Bu hatayı alıyorum: [hata mesajı]
Dosya: [dosya_adı.py]
Ne yapmalıyım? Basit çözüm öner."
```

### Kod İnceleme
```
"[dosya_adı.py] dosyasını oku ve şunları kontrol et:
1. Bir hata var mı?
2. Basitleştirilebilecek yer var mı?
3. Eksik bir şey var mı?"
```

### Test Yazma
```
"[dosya_adı.py]'deki [fonksiyon_adı] fonksiyonunu test etmek
için scripts/ altına basit bir test scripti yaz."
```
