###                       _--------------testing--------------------------
c_lambda=0.6
cur_step = 0
generator_losses = []
critic_losses = []
n_epochs=500
crit_repeats=3
display_step=3
for epoch in range(n_epochs):
    # Dataloader returns the batches
    for i in range(500):
        s_r_path=os.path.join('images/concatenated/real/img{}.jpg'.format(i))
        c_path=os.path.join('images/concatenated/crap/img{}.jpg'.format(i))
        super_res=Image.open(s_r_path)
        super_res=transforms.ToTensor()(super_res)
        super_res=super_res.unsqueeze(0)
        print(super_res.shape)
        crap=Image.open(c_path)
        crap=transforms.ToTensor()(crap)
        crap=crap.unsqueeze(0)
        cur_batch_size = len(super_res)
        super_res = super_res.to(device)

        mean_iteration_critic_loss = 0
        for _ in range(crit_repeats):
            ### Update critic ###
            crit_opt.zero_grad()
            fake = gen(crap)
            crit_fake_pred = crit(fake.detach())
            print(crit_fake_pred.shape)
            crit_real_pred = crit(super_res)

            epsilon = torch.rand(len(super_res), 1, 1, 1, device=device, requires_grad=True)
            gradient = get_gradient(crit, super_res, fake.detach(), epsilon)
            gp = gradient_penalty(gradient)
            crit_loss = get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda)

            # Keep track of the average critic loss in this batch
            mean_iteration_critic_loss += crit_loss.item() / crit_repeats
            # Update gradients
            crit_loss.backward(retain_graph=True)
            # Update optimizer
            crit_opt.step()
        critic_losses += [mean_iteration_critic_loss]

        ### Update generator ###
        gen_opt.zero_grad()
        fake_2 = gen(crap)
        crit_fake_pred = crit(fake_2)
        
        gen_loss = get_gen_loss(crit_fake_pred)
        gen_loss.backward()

        # Update the weights
        gen_opt.step()

        # Keep track of the average generator loss
        generator_losses += [gen_loss.item()]

        ### Visualization code ###
        if cur_step % display_step == 0 and cur_step > 0:
            gen_mean = sum(generator_losses[-display_step:]) / display_step
            crit_mean = sum(critic_losses[-display_step:]) / display_step
            print(f"Step {cur_step}: Generator loss: {gen_mean}, critic loss: {crit_mean}")
            show_tensor_images(fake_2)
            show_tensor_images(super_res)
            step_bins = 20
            num_examples = (len(generator_losses) // step_bins) * step_bins
            plt.plot(
                range(num_examples // step_bins), 
                torch.Tensor(generator_losses[:num_examples]).view(-1, step_bins).mean(1),
                label="Generator Loss"
            )
            plt.plot(
                range(num_examples // step_bins), 
                torch.Tensor(critic_losses[:num_examples]).view(-1, step_bins).mean(1),
                label="Critic Loss"
            )
            plt.legend()
            plt.show()

        cur_step += 1