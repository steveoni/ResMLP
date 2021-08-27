def train(epochs,trainloader, testloader, costFunc, model, device, optimizer):
  for epoch in range(epochs):
    train_loss = 0.
    valid_loss = 0.
    model.train()
    for data, labels in trainloader:
      data,labels = data.to(device),labels.to(device)
      prediction = model(data)
      loss = costFunc(prediction,labels)
      train_loss += loss.item()
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    
    model.eval()
    for data, labels in testloader:
      data,labels = data.to(device),labels.to(device)
      prediction = model(data)
      loss = costFunc(prediction,labels)
      valid_loss += loss.item()

    print(f'Epoch {epoch} Training Loss: {train_loss /len(trainloader)} ---- \ valid loss: {valid_loss / len(testloader)}')
