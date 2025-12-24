# final_training_fixed.py
import torch
import torch.nn as nn
#import torch.nn.functional as F
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, random_split, Subset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models
import numpy as np
from sklearn.metrics import precision_score, recall_score
import os
import time

# ========== ДИАГНОСТИКА GPU ==========
print("=" * 60)
print("🤖 ДИАГНОСТИКА СИСТЕМЫ GOOGLE COLAB")
print("=" * 60)

# Проверка GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ GPU обнаружена: {torch.cuda.get_device_name(0)}")
    print(f"   Память: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"   CUDA: {torch.version.cuda}")
else:
    device = torch.device("cpu")
    print("❌ GPU не обнаружена, используется CPU")

print(f"⚡ Используется устройство: {device}")
print("-" * 60)

class Bottleneck(nn.Module):
    """Bottleneck block для ResNet50/101/152"""
    expansion = 4  # Расширение каналов в 4 раза

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        # 1x1 свёртка для уменьшения размерности
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3x3 свёртка
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 свёртка для увеличения размерности
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # Residual connection
        out = self.relu(out)

        return out


class ResNet50(nn.Module):
    """Реализация ResNet50 с нуля"""

    def __init__(self, num_classes=120, zero_init_residual=False):
        super(ResNet50, self).__init__()

        # Параметры ResNet50
        self.in_channels = 64

        # Начальные слои
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet слои
        self.layer1 = self._make_layer(Bottleneck, 64, 3, stride=1)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

        # Классификатор
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        # Инициализация весов
        self._initialize_weights(zero_init_residual)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """Создание слоя ResNet"""
        downsample = None

        # Если нужно изменить размеры (stride != 1 или каналы не совпадают)
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        # Первый блок в слое может иметь downsample
        layers.append(block(self.in_channels, out_channels, stride, downsample))

        self.in_channels = out_channels * block.expansion

        # Остальные блоки
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self, zero_init_residual):
        """Инициализация весов как в оригинальной ResNet"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize последний BatchNorm в каждом residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x):
        # Начальные слои
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ResNet слои
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Классификация
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def setup_environment():
    # В Colab датасет будет в /content после скачивания
    # На локальном устройстве путь к датасету такой: "C:/Users/1/.cache/kagglehub/datasets/jessicali9530/stanford-dogs-dataset/versions/2/images"
    data_dir = "/content/dogs"  # Путь после распаковки
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Проверка использования GPU
    if torch.cuda.is_available():
        print(f"📊 Память GPU до загрузки: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print(f"Устройство: {device}")
    return device, data_dir


def prepare_dataloaders_smart(data_dir, batch_size=256, max_images=12000):
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # Загружаем ОДИН датасет без трансформаций
    full_dataset = ImageFolder(
        root=os.path.join(data_dir, 'Images'),
        transform=None  # Пока без трансформаций
    )

    # Ограничиваем количество
    if len(full_dataset) > max_images:
        indices = torch.randperm(len(full_dataset))[:max_images]
        dataset = Subset(full_dataset, indices)
        print(f"  Используется {max_images} изображений")
    else:
        dataset = full_dataset

    # Разделяем на train/val/test
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    # ВАЖНО: seed для воспроизводимости
    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices, test_indices = random_split(
        range(total_size),
        [train_size, val_size, test_size],
        generator=generator
    )

    # Создаем разные трансформации
    transform_train = transforms.Compose([
        transforms.Resize(128),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    transform_val_test = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    # Создаем подмножества с разными трансформациями
    class TransformedSubset(Subset):
        """Подмножество с своей трансформацией"""

        def __init__(self, dataset, indices, transform=None):
            super().__init__(dataset, indices)
            self.transform = transform

        def __getitem__(self, idx):
            x, y = self.dataset[self.indices[idx]]
            if self.transform:
                x = self.transform(x)
            return x, y

    train_dataset = TransformedSubset(full_dataset, train_indices, transform_train)
    val_dataset = TransformedSubset(full_dataset, val_indices, transform_val_test)
    test_dataset = TransformedSubset(full_dataset, test_indices, transform_val_test)

    # Создаем DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    print(f"Классов: {len(full_dataset.classes)}")
    print(f"Данные: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    return train_loader, val_loader, test_loader, full_dataset.classes


def create_model(pretrained=True, num_classes=120, device=None):
    """Создание модели с проверкой устройства"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n🔧 Создание модели на {device}")

    if pretrained:
        from torchvision.models import ResNet50_Weights
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        print(" Предобученная ResNet50")
    else:
        #model = models.resnet50(weights=None)
        # Реализация ResNet50 с нуля
        model = ResNet50(num_classes = num_classes)
        print(" ResNet50 с нуля")

    # Перенос модели на устройство
    model = model.to(device)

    # Проверка
    print(f"📏 Параметров модели: {sum(p.numel() for p in model.parameters()):,}")
    print(f"📍 Модель на устройстве: {next(model.parameters()).device}")

    return model


def calculate_metrics(model, data_loader, device):
    """Расчет метрик"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)

    return accuracy, precision, recall


def train_model_unified(model, train_loader, val_loader, device,
                        use_swa=False, num_epochs=30, is_pretrained=True):

    criterion = nn.CrossEntropyLoss().to(device)

    if is_pretrained:
        print("=" * 60)
        print("РЕЖИМ: Fine-tuning предобученной модели")
        print("   Первые слои (conv1, layer1, layer2) заморожены")
        print("=" * 60)

        # === ШАГ 1: Замораживаем ВСЕ слои ===
        for param in model.parameters():
            param.requires_grad = False

        # === ФАЗА 1: Обучение только классификатора (5 эпох) ===
        print("\nФаза 1: Обучение только классификатора (5 эпох)")
        print("   Обучается: fc (классификатор)")
        print("   Все остальные слои заморожены")

        # Размораживаем ТОЛЬКО классификатор
        for param in model.fc.parameters():
            param.requires_grad = True

        optimizer = optim.Adam(model.fc.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

        for epoch in range(5):
            model.train()
            total_loss = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            val_acc, _, _ = calculate_metrics(model, val_loader, device)
            print(f"   Эпоха {epoch + 1:2d}: "
                  f"Loss={total_loss / len(train_loader):.4f}, "
                  f"Val Acc={val_acc:.4f}")

        # === ФАЗА 2: Размораживаем layer4 (10 эпох) ===
        print("\nФаза 2: Размораживаем layer4 (10 эпох)")
        print("   Обучаются: layer4, fc")
        print("   Заморожены: conv1, layer1, layer2, layer3")

        # Размораживаем ТОЛЬКО layer4
        for param in model.layer4.parameters():
            param.requires_grad = True

        # Оптимизатор для layer4 + fc
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(trainable_params, lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

        for epoch in range(10):
            model.train()
            total_loss = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            val_acc, _, _ = calculate_metrics(model, val_loader, device)
            print(f"   Эпоха {epoch + 6:2d}: "
                  f"Loss={total_loss / len(train_loader):.4f}, "
                  f"Val Acc={val_acc:.4f}")

        # === ФАЗА 3: Размораживаем layer3 (если данных достаточно) ===
        dataset_size = len(train_loader.dataset)
        remaining_epochs = num_epochs - 15

        if dataset_size > 10000:  # Для Stanford Dogs (20k) - размораживаем
            print(f"\nФаза 3: Размораживаем layer3 ({remaining_epochs} эпох)")
            print("   Обучаются: layer3, layer4, fc")
            print("   Заморожены: conv1, layer1, layer2")

            # Размораживаем layer3
            for param in model.layer3.parameters():
                param.requires_grad = True

            # Новый оптимизатор для layer3 + layer4 + fc
            trainable_params = filter(lambda p: p.requires_grad, model.parameters())
            optimizer = optim.Adam(trainable_params, lr=0.0001)  # Меньший LR для layer3
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        else:
            print(f"\nФаза 3: Продолжаем обучение layer4 + fc ({remaining_epochs} эпох)")
            print("   Обучаются: layer4, fc")
            print("   Заморожены: conv1, layer1, layer2, layer3")
            # Оптимизатор остаётся тот же (только layer4 + fc)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    else:  # Обучение с нуля
        print("=" * 50)
        print("РЕЖИМ: Обучение с нуля")
        print("=" * 50)

        for param in model.parameters():
            param.requires_grad = True

        optimizer = optim.Adam(model.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        remaining_epochs = num_epochs

    # === ОБЩАЯ ФАЗА ОБУЧЕНИЯ ===
    print(f"\nОсновное обучение ({remaining_epochs} эпох)")

    # Показываем какие слои обучаются
    trainable_layers = []
    for name, param in model.named_parameters():
        if param.requires_grad and name.split('.')[0] not in trainable_layers:
            trainable_layers.append(name.split('.')[0])

    print(f"   Обучаемые слои: {', '.join(sorted(set(trainable_layers)))}")

    # Настройка SWA
    swa_model = None
    if use_swa:
        swa_model = AveragedModel(model).to(device)
        swa_start = int(remaining_epochs * 0.7)  # Начинаем с 70% эпох
        print(f"   SWA активируется с эпохи {swa_start}")

    for epoch in range(remaining_epochs):
        model.train()
        total_loss = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # SWA обновление
        if use_swa and swa_model and (epoch >= swa_start):
            swa_model.update_parameters(model)

        # Обновление LR
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Валидация
        val_acc, _, _ = calculate_metrics(model, val_loader, device)

        print(f"   Эпоха {epoch + 1:3d}: "
              f"Loss={total_loss / len(train_loader):.4f}, "
              f"Val Acc={val_acc:.4f}, "
              f"LR={current_lr:.6f}")

    # Финальная обработка SWA
    if use_swa and swa_model:
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        print(" SWA модель готова")
        return swa_model

    return model


def main():
    print("=" * 60)
    print(" СРАВНЕНИЕ МЕТОДОВ ОБУЧЕНИЯ - Stanford Dogs")
    print("=" * 60)

    total_start = time.time()
    device, data_dir = setup_environment()
    BATCH_SIZE = 256 # Для GPU нормально

    train_loader, val_loader, test_loader, classes = prepare_dataloaders_smart(data_dir, batch_size=BATCH_SIZE, max_images=12000)

    results = {}

    # ЭКСПЕРИМЕНТ 1: Предобученная + Adam (10 эпох)
    print("\n" + "=" * 50)
    print(" ЭКСПЕРИМЕНТ 1: Предобученная + Adam")
    print("=" * 50)

    exp1_start = time.time()
    model1 = create_model(pretrained=True, num_classes=120, device=device)
    model1 = train_model_unified(model1, train_loader, val_loader, device,
                                 use_swa=False, num_epochs=30, is_pretrained=True)

    test_acc1, test_prec1, test_rec1 = calculate_metrics(model1, test_loader, device)
    results['Pretrained+Adam'] = {'accuracy': test_acc1, 'precision': test_prec1, 'recall': test_rec1}

    # ЭКСПЕРИМЕНТ 2: Предобученная + Averaged Adam (10 эпох)
    print("\n" + "=" * 50)
    print(" ЭКСПЕРИМЕНТ 2: Предобученная + Averaged Adam")
    print("=" * 50)

    exp2_start = time.time()
    model2 = create_model(pretrained=True)
    model2 = train_model_unified(model2, train_loader, val_loader, device,
                                 use_swa=True, num_epochs=10, is_pretrained=True)

    test_acc2, test_prec2, test_rec2 = calculate_metrics(model2, test_loader, device)
    results['Pretrained+AveragedAdam'] = {'accuracy': test_acc2, 'precision': test_prec2, 'recall': test_rec2}
    #
    # ЭКСПЕРИМЕНТ 3: С нуля + Adam (10 эпох)
    # print("\n" + "=" * 50)
    # print(" ЭКСПЕРИМЕНТ 3: Обучение с нуля + Adam")
    # print("=" * 50)
    # print(" Примечание: ResNet50 с нуля требует 50+ эпох")
    #
    # exp3_start = time.time()
    # model3 = create_model(pretrained=False)
    # model3 = train_model_unified(model3, train_loader, val_loader, device,
    #                              use_swa=False, num_epochs=10, is_pretrained=False)
    #
    # test_acc3, test_prec3, test_rec3 = calculate_metrics(model3, test_loader, device)
    # results['Scratch+Adam'] = {'accuracy': test_acc3, 'precision': test_prec3, 'recall': test_rec3}

    # РЕЗУЛЬТАТЫ
    print("\n" + "=" * 60)
    print(" ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("=" * 60)

    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")

    total_time = (time.time() - total_start) / 60
    print(f"\n ОБУЧЕНИЕ ЗАВЕРШЕНО ЗА {total_time:.1f} МИНУТ")

    # Очистка памяти GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"\n🧹 Память GPU очищена")


if __name__ == "__main__":
    main()