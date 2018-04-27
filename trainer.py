from torch.autograd import Variable
import numpy as np


def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, start_epoch=0):
    print(len(train_loader))
    print(len(val_loader))
    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train phase
        train_loss = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)

        val_loss = test_epoch(val_loader, model, loss_fn, cuda)
        val_loss /= len(val_loader)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        print(message)


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval):

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, data in enumerate(train_loader):
        if cuda:
            data = tuple(d.cuda() for d in data)
        data = tuple(Variable(d) for d in data)

        optimizer.zero_grad()
        outputs = model(*data)

        # if type(outputs) not in (tuple, list):
        #     outputs = (outputs,)

        loss_inputs = outputs
        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.data[0])
        total_loss += loss.data[0]
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss


def test_epoch(val_loader, model, loss_fn, cuda):

    model.eval()
    val_loss = 0
    for batch_idx, data in enumerate(val_loader):
        if cuda:
            data = tuple(d.cuda() for d in data)
        data = tuple(Variable(d, volatile=True) for d in data)
        outputs = model(*data)
        # if type(outputs) not in (tuple, list):
        #     outputs = (outputs,)
        loss_inputs = outputs
        # if target is not None:
        #     target = Variable(target, volatile=True)
        #     target = (target,)
        #     loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        val_loss += loss.data[0]

    return val_loss
