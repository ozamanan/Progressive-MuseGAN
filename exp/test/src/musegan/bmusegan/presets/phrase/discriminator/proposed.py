"""Network architecture of the proposed discriminator
"""
NET_D = {}

NET_D['shared'] = [
    ('conv3d', (8, (1, 1, 7), (1, 1, 7)), None, 'lrelu'),
    ('conv3d', (8, (1, 1, 3), (1, 1, 3)), None, 'lrelu'),
    ('conv3d', (8, (1, 1, 2), (1, 1, 2)), None, 'lrelu'),
    ('conv3d', (8, (1, 1, 2), (1, 1, 2)), None, 'lrelu'),
    ('conv3d', (8, (1, 2, 1), (1, 2, 1)), None, 'lrelu'),
    ('conv3d', (8, (1, 2, 1), (1, 2, 1)), None, 'lrelu'),
    ('conv3d', (8, (1, 2, 1), (1, 2, 1)), None, 'lrelu'),
    ('conv3d', (8, (1, 2, 1), (1, 2, 1)), None, 'lrelu'),
    ('conv3d', (8, (1, 2, 1), (1, 2, 1)), None, 'lrelu'),
    ('conv3d', (8, (1, 2, 1), (1, 2, 1)), None, 'lrelu'),
    ('reshape', (4*8)),
    ('dense', 1),
]
