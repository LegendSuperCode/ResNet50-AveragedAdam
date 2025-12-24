import torch
import torch.nn as nn
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
from torch.utils.data import Dataset, DataLoader, random_split, Subset
# В начале файла, после импортов:
from torch.amp import autocast, GradScaler

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
                if m.bias is not None:
                  nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Для выходного слоя std больше
                if m is self.fc:
                    nn.init.normal_(m.weight, 0, 0.1)  # std=0.1 для 120 классов
                else:
                    nn.init.normal_(m.weight, 0, 0.01)  # std=0.01 для других Linear
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
    # Трансформации
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # Для модели ResNet50 с нуля поменяли Resize с 256 до 512
    # Кроме того, поменяли CenterCrop с 224 на 448 для этой же необученной модели
    # Для обученных моделей Resize и CenterCrop были другими
    transform_train = transforms.Compose([
        transforms.Resize(512),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(5),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    # Загружаем базовый датасет
    base_dataset = ImageFolder(os.path.join(data_dir, 'Images'))

    # Ограничиваем
    if len(base_dataset) > max_images:
        indices = torch.randperm(len(base_dataset))[:max_images]
        base_dataset = Subset(base_dataset, indices)

    # Разделяем индексы
    train_size = int(0.7 * len(base_dataset))
    val_size = int(0.15 * len(base_dataset))
    test_size = len(base_dataset) - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices, test_indices = random_split(
        range(len(base_dataset)),
        [train_size, val_size, test_size],
        generator=generator
    )

    # Кастомный DatasetWrapper
    class DatasetWrapper(Dataset):
        def __init__(self, base_dataset, indices, transform):
            self.base_dataset = base_dataset
            self.indices = list(indices)
            self.transform = transform

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            real_idx = self.indices[idx]
            img, label = self.base_dataset[real_idx]
            if self.transform:
                img = self.transform(img)
            return img, label

    # Создаем три независимых датасета
    train_dataset = DatasetWrapper(base_dataset, train_indices.indices, transform_train)
    val_dataset = DatasetWrapper(base_dataset, val_indices.indices, transform_test)
    test_dataset = DatasetWrapper(base_dataset, test_indices.indices, transform_test)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers = True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers = True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers = True)

    return train_loader, val_loader, test_loader, base_dataset.dataset.classes


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
    """
    Упрощенная версия с правильным fine-tuning и Mixed Precision
    Только StepLR, без косинусного аннигинга
    Первые слои (conv1, layer1, layer2) заморожены для предобученных моделей
    """

    criterion = nn.CrossEntropyLoss().to(device)

    # Проверьте:
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Batch size: {train_loader.batch_size}")

    # Mixed Precision Training для ускорения GPU
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    print(f"   Mixed Precision: {'ВКЛЮЧЕНО' if scaler else 'ОТКЛЮЧЕНО (нет GPU)'}")

    if is_pretrained:
        print("=" * 60)
        print("РЕЖИМ: Fine-tuning предобученной модели")
        print("   Первые слои (conv1, layer1, layer2) заморожены")
        print("=" * 60)

        # === ШАГ 1: Замораживаем ВСЕ слои ===
        for param in model.parameters():
            param.requires_grad = False

        # === ФАЗА 1: Обучение только классификатора (8 эпох) ===
        print("\nФаза 1: Обучение только классификатора (8 эпох)")
        print("   Обучается: fc (классификатор)")
        print("   Все остальные слои заморожены")

        # Размораживаем ТОЛЬКО классификатор
        for param in model.fc.parameters():
            param.requires_grad = True

        optimizer = optim.Adam(model.fc.parameters(), lr=0.002, weight_decay=0.01)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.65)

        for epoch in range(8):
            model.train()
            total_loss = 0
            epoch_start = time.time()

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad()

                # Mixed precision forward
                if scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    # Mixed precision backward
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()

            scheduler.step()
            epoch_time = time.time() - epoch_start
            val_acc, _, _ = calculate_metrics(model, val_loader, device)

            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1e9
                print(f"   Эпоха {epoch+1:2d}: "
                      f"Loss={total_loss/len(train_loader):.4f}, "
                      f"Val Acc={val_acc:.4f}, "
                      f"Время={epoch_time:.1f}с, "
                      f"GPU={memory_used:.2f}GB")
            else:
                print(f"   Эпоха {epoch+1:2d}: "
                      f"Loss={total_loss/len(train_loader):.4f}, "
                      f"Val Acc={val_acc:.4f}, "
                      f"Время={epoch_time:.1f}с")

        # === ФАЗА 2: Размораживаем layer4 (10 эпох) ===
        print("\nФаза 2: Размораживаем layer4 (10 эпох)")
        print("   Обучаются: layer4, fc")
        print("   Заморожены: conv1, layer1, layer2, layer3")

        # Размораживаем ТОЛЬКО layer4
        for param in model.layer4.parameters():
            param.requires_grad = True

        # Оптимизатор для layer4 + fc
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam([
            {'params': model.fc.parameters(), 'lr': 0.0001},
            {'params': model.layer4.parameters(), 'lr': 0.00002}  # В 5 раз меньше!
        ], weight_decay=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.75)

        for epoch in range(10):
            model.train()
            total_loss = 0
            epoch_start = time.time()

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad()

                # Mixed precision
                if scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()

            scheduler.step()
            epoch_time = time.time() - epoch_start
            val_acc, _, _ = calculate_metrics(model, val_loader, device)

            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1e9
                print(f"   Эпоха {epoch+9:2d}: "
                      f"Loss={total_loss/len(train_loader):.4f}, "
                      f"Val Acc={val_acc:.4f}, "
                      f"Время={epoch_time:.1f}с, "
                      f"GPU={memory_used:.2f}GB")
            else:
                print(f"   Эпоха {epoch+9:2d}: "
                      f"Loss={total_loss/len(train_loader):.4f}, "
                      f"Val Acc={val_acc:.4f}, "
                      f"Время={epoch_time:.1f}с")

        # === ФАЗА 3: Размораживаем layer3 (если данных достаточно) ===
        dataset_size = len(train_loader.dataset.base_dataset)
        remaining_epochs = num_epochs - 18

        if dataset_size > 10000:  # Для Stanford Dogs (20k) - размораживаем
            print(f"\nФаза 3: Размораживаем layer3 ({remaining_epochs} эпох)")
            print("   Обучаются: layer3, layer4, fc")
            print("   Заморожены: conv1, layer1, layer2")

            # Размораживаем layer3
            for param in model.layer3.parameters():
                param.requires_grad = True

            # Новый оптимизатор для layer3 + layer4 + fc
            trainable_params = filter(lambda p: p.requires_grad, model.parameters())
            optimizer = optim.Adam(trainable_params, lr=0.0001, weight_decay=0.01)  # Меньший LR для layer3
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.85)

        else:
            print(f"\nФаза 3: Продолжаем обучение layer4 + fc ({remaining_epochs} эпох)")
            print("   Обучаются: layer4, fc")
            print("   Заморожены: conv1, layer1, layer2, layer3")
            # Оптимизатор остаётся тот же (только layer4 + fc)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.85)

    else:  # Обучение с нуля
        print("=" * 50)
        print("РЕЖИМ: Обучение с нуля")
        print("=" * 50)

        for param in model.parameters():
            param.requires_grad = True

        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
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
        epoch_start = time.time()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()

            # Mixed precision
            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        epoch_time = time.time() - epoch_start

        # SWA обновление
        if use_swa and swa_model and (epoch >= swa_start):
            swa_model.update_parameters(model)

        # Обновление LR
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Валидация
        val_acc, _, _ = calculate_metrics(model, val_loader, device)

        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1e9
            print(f"   Эпоха {epoch+1:3d}: "
                  f"Loss={total_loss/len(train_loader):.4f}, "
                  f"Val Acc={val_acc:.4f}, "
                  f"LR={current_lr:.6f}, "
                  f"Время={epoch_time:.1f}с, "
                  f"GPU={memory_used:.2f}GB")
        else:
            print(f"   Эпоха {epoch+1:3d}: "
                  f"Loss={total_loss/len(train_loader):.4f}, "
                  f"Val Acc={val_acc:.4f}, "
                  f"LR={current_lr:.6f}, "
                  f"Время={epoch_time:.1f}с")

    # Финальная обработка SWA
    if use_swa and swa_model:
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        print("✅ SWA модель готова")
        return swa_model

    return model


def main():
    print("=" * 60)
    print(" СРАВНЕНИЕ МЕТОДОВ ОБУЧЕНИЯ - Stanford Dogs")
    print("=" * 60)

    total_start = time.time()
    device, data_dir = setup_environment()
    BATCH_SIZE = 64 # Для необученной модели нормально

    train_loader, val_loader, test_loader, classes = prepare_dataloaders_smart(data_dir, batch_size=BATCH_SIZE, max_images=12000)

    #ЭКСПЕРИМЕНТ 1: Предобученная + Adam (10 эпох)
    print("\n" + "=" * 50)
    print(" ЭКСПЕРИМЕНТ 1: Предобученная + Adam")
    print("=" * 50)

    exp1_start = time.time()
    model1 = create_model(pretrained=True, num_classes=120, device=device)
    model1 = train_model_unified(model1, train_loader, val_loader, device,
                                 use_swa=False, num_epochs=50, is_pretrained=True)

    # Train метрики
    print("\n📊 Train выборка:")
    train_acc1, train_prec1, train_rec1 = calculate_metrics(model1, train_loader, device)
    print(f"  Accuracy:  {train_acc1:.4f}")
    print(f"  Precision: {train_prec1:.4f}")
    print(f"  Recall:    {train_rec1:.4f}")

    # Validation метрики
    print("\n📊 Validation выборка:")
    val_acc1, val_prec1, val_rec1 = calculate_metrics(model1, val_loader, device)
    print(f"  Accuracy:  {val_acc1:.4f}")
    print(f"  Precision: {val_prec1:.4f}")
    print(f"  Recall:    {val_rec1:.4f}")

    # Test метрики
    print("\n📊 Test выборка:")
    test_acc1, test_prec1, test_rec1 = calculate_metrics(model1, test_loader, device)
    print(f"  Accuracy:  {test_acc1:.4f}")
    print(f"  Precision: {test_prec1:.4f}")
    print(f"  Recall:    {test_rec1:.4f}")

    #ЭКСПЕРИМЕНТ 2: Предобученная + Averaged Adam (10 эпох)
    print("\n" + "=" * 50)
    print(" ЭКСПЕРИМЕНТ 2: Предобученная + Averaged Adam")
    print("=" * 50)

    exp2_start = time.time()
    model2 = create_model(pretrained=True)
    model2 = train_model_unified(model2, train_loader, val_loader, device,
                                 use_swa=True, num_epochs=50, is_pretrained=True)

    # Train метрики
    print("\n📊 Train выборка:")
    train_acc2, train_prec2, train_rec2 = calculate_metrics(model2, train_loader, device)
    print(f"  Accuracy:  {train_acc2:.4f}")
    print(f"  Precision: {train_prec2:.4f}")
    print(f"  Recall:    {train_rec2:.4f}")

    # Validation метрики
    print("\n📊 Validation выборка:")
    val_acc2, val_prec2, val_rec2 = calculate_metrics(model2, val_loader, device)
    print(f"  Accuracy:  {val_acc2:.4f}")
    print(f"  Precision: {val_prec2:.4f}")
    print(f"  Recall:    {val_rec2:.4f}")

    # Test метрики
    print("\n📊 Test выборка:")
    test_acc2, test_prec2, test_rec2 = calculate_metrics(model2, test_loader, device)
    print(f"  Accuracy:  {test_acc2:.4f}")
    print(f"  Precision: {test_prec2:.4f}")
    print(f"  Recall:    {test_rec2:.4f}")

    #ЭКСПЕРИМЕНТ 3: С нуля + Adam (10 эпох)
    print("\n" + "=" * 50)
    print(" ЭКСПЕРИМЕНТ 3: Обучение с нуля + Adam")
    print("=" * 50)
    print(" Примечание: ResNet50 с нуля требует 50+ эпох")

    exp3_start = time.time()
    model3 = create_model(pretrained=False)
    model3 = train_model_unified(model3, train_loader, val_loader, device,
                                 use_swa=False, num_epochs=80, is_pretrained=False)

    # Train метрики
    print("\n📊 Train выборка:")
    train_acc3, train_prec3, train_rec3 = calculate_metrics(model3, train_loader, device)
    print(f"  Accuracy:  {train_acc3:.4f}")
    print(f"  Precision: {train_prec3:.4f}")
    print(f"  Recall:    {train_rec3:.4f}")

    # Validation метрики
    print("\n📊 Validation выборка:")
    val_acc3, val_prec3, val_rec3 = calculate_metrics(model3, val_loader, device)
    print(f"  Accuracy:  {val_acc3:.4f}")
    print(f"  Precision: {val_prec3:.4f}")
    print(f"  Recall:    {val_rec3:.4f}")

    # Test метрики
    print("\n📊 Test выборка:")
    test_acc3, test_prec3, test_rec3 = calculate_metrics(model3, test_loader, device)
    print(f"  Accuracy:  {test_acc3:.4f}")
    print(f"  Precision: {test_prec3:.4f}")
    print(f"  Recall:    {test_rec3:.4f}")

    total_time = (time.time() - total_start) / 60
    print(f"\n ОБУЧЕНИЕ ЗАВЕРШЕНО ЗА {total_time:.1f} МИНУТ")

    # Очистка памяти GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"\n🧹 Память GPU очищена")


if __name__ == "__main__":
    main()