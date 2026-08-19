const menu = document.querySelector('.docs-menu');
const sidebar = document.querySelector('.docs-sidebar');

if (menu && sidebar) {
  menu.addEventListener('click', () => {
    const open = document.body.classList.toggle('nav-open');
    menu.setAttribute('aria-expanded', String(open));
  });
  sidebar.addEventListener('click', (event) => {
    if (event.target.closest('a')) {
      document.body.classList.remove('nav-open');
      menu.setAttribute('aria-expanded', 'false');
    }
  });
}

document.querySelectorAll('.markdown-body pre').forEach((pre) => {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'copy-code';
  button.textContent = 'COPY';
  button.addEventListener('click', async () => {
    await navigator.clipboard.writeText(pre.innerText.replace(/^COPY\n?/, ''));
    button.textContent = 'COPIED';
    window.setTimeout(() => { button.textContent = 'COPY'; }, 1200);
  });
  pre.appendChild(button);
});
