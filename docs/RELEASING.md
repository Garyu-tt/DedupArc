# Выпуск новой версии

1. Изменить `internal/version/version.go`.
2. Добавить запись в `CHANGELOG.md`.
3. Запустить:

```powershell
go test ./...
go vet ./...
.\scripts\build-windows.ps1
```

Скрипт берёт номер версии из `internal/version/version.go`, поэтому launcher, updater, setup и имена release-файлов не должны расходиться по версиям.

4. Проверить установку поверх предыдущей версии, ассоциацию `.darc`, создание архива, открытие, запуск файла из архива, распаковку и удаление.
5. Для изменений формата отдельно проверить чтение архива предыдущей версии.
6. Убедиться, что GitHub Actions secret `DEDUPARC_RELEASE_KEY` настроен.
7. Создать тег `vX.Y.Z` и отправить его в GitHub.
8. Workflow `release.yml` соберёт подписанный update-пакет и создаст GitHub Release.

Приватный Ed25519-ключ не добавляется в git ни в каком виде.
